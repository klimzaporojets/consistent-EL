# (kzaporoj 26/11/2020) - unlike data_reader.py which works on specific inputs from DWIE dataset
# (e.g. json files like DW_x.json), this reader is designed to take input from the user. In the initial version,
# this input is just a plain text like "Germany has seen more than 1 million cases of COVID-19 since the pandemic began".
# In principle the input can be one sentence or a document composed by multiple sentences. The goal of this module is to
#   1. Tokenize the input text
#   2. Assign token ids to each of the tokens (based on the respective dictionary)
#   3. Extract all possible spans from the text
#   4. Assign candidate links to these spans
#   5. Assign candidate ids to these spans (based on the respective entity dictionary)
#   6. .... others (TODO: finish this list).
# However, unlike cpn.data_reader.DatasetCPN#convert, it directly doesn't have access to "mentions" or "concepts",
# which makes this reader perfect for evaluating the dataset just on plain text (using services such as GERBIL).
# It uses some of the useful functions from data_reader.py . In other words, this module can be seen as a simplified
# data_reader.py which works on plain input text that will be given by any user.
import torch

from cpn.data_reader import AbstractDataReader
from cpn.tokenizer import TokenizerCPN
from datass.dictionary import Dictionary
from datass.transform import get_token_buckets
from modules.ner.spanner import create_all_spans
from modules.utils.misc import indices_to_spans


class DataReaderAPI(AbstractDataReader):
    def __init__(self, config, dictionaries, linking_candidates=None):
        # AbstractDataReader.__init__(self)
        super().__init__()
        self.dict_words: Dictionary = dictionaries['words']
        self.dict_characters: Dictionary = dictionaries['characters']
        self.dict_whitespace: Dictionary = dictionaries.get('whitespace', None)
        self.dict_tags: Dictionary = dictionaries['tags']
        self.dict_relations: Dictionary = dictionaries['relations']
        self.dict_entities: Dictionary = dictionaries.get('entities', None)
        self.linking_candidates = linking_candidates
        self.max_span_length = config['model']['max_span_length']
        self.include_nill_in_candidates = config['dataloader']['include_nill_in_candidates']
        self.include_none_in_candidates = config['dataloader']['include_none_in_candidates']
        self.all_spans_candidates = config['dataloader']['all_spans_candidates']
        self.shuffle_candidates = config['datasets']['test']['shuffle_candidates']
        self.max_link_candidates = config['datasets']['test'].get('max_link_candidates', None)

        self.tokenizer = TokenizerCPN()

    def convert(self, data):
        """

        :param data: initially has only 'text' with the plain text to be processed
        :return:
        """
        input_text = data['text']

        tokens = self.tokenizer.tokenize(input_text)

        begin = [token['offset'] for token in tokens]
        end = [token['offset'] + token['length'] for token in tokens]
        tokens = [token['token'] for token in tokens]

        n_tokens = len(tokens)

        if n_tokens == 0:
            print("WARNING: (data_reader_api) dropping empty document")
            return

        # begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
        # end_to_index = {pos: idx for idx, pos in enumerate(end)}

        # all_doc_candidates_no_nill = list()

        span_begin, span_end = create_all_spans(1, n_tokens, self.max_span_length)

        lengths = torch.tensor([n_tokens], dtype=torch.long)
        span_mask = (span_end < lengths.unsqueeze(-1).unsqueeze(-1)).float()

        nr_possible_spans = (span_mask.size(-1) * span_mask.size(-2))

        span_masked_scores = span_mask.view(span_mask.size(0), -1)

        # top_indices_sorted = torch.range(0, nr_possible_spans - 1, dtype=torch.int32).unsqueeze(0)
        top_indices_sorted = torch.arange(0, nr_possible_spans, dtype=torch.int32).unsqueeze(0)

        all_possible_spans = indices_to_spans(top_indices_sorted, torch.tensor([nr_possible_spans],
                                                                               dtype=torch.int),
                                              self.max_span_length)[0]

        linker_cands_all_spans, linker_cands_all_spans_scores = \
            self.get_linker_candidates_all_spans(input_text, None, all_spans=all_possible_spans, begin=begin, end=end,
                                                 span_mask=span_masked_scores[0])

        # linker_targets_all_spans = self.get_linker_targets_all_spans(data, all_possible_spans,
        #                                                              linker_cands_all_spans)

        linker_candidates = linker_cands_all_spans
        # linker_targets = linker_targets_all_spans
        linker_scores = linker_cands_all_spans_scores
        # here tokens should be of the following format:
        #   <class 'list'>: ['Healthy', 'Travel', '-', 'Dr', '.', 'Thorsten', 'Onno', 'Bender', 'in', 'our', 'studio', 'The', 'specialist', 'for', 'flight', 'medicine', 'and', 'internal', 'medicine', 'provides', 'you', 'with', 'tips', 'on', 'tackling', 'motion', 'sickness', 'and', 'jet', 'lag', 'and', 'tells', 'us', 'which', 'pre', '-', 'holiday', 'vaccinations', 'make', 'sense', '.', 'Flight', 'Medicine', 'Unit', ',', 'Charité', 'University', 'Hospital', ',', 'Berlin', 'Campus', 'Virchow', 'Clinic', 'Augustenburger', 'Platz', '1', '13353', 'Berlin', 'http', ':', '//', 'www', '.', 'fliegerarzt', '-', 'berlin', '.', 'de', '/']
        # token_indices should be of the following format:
        #   <class 'list'>: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 16, 19, 20, 21, 22, 23, 24, 25, 26, 17, 27, 28, 17, 29, 30, 31, 32, 3, 33, 34, 35, 36, 5, 37, 38, 39, 40, 41, 42, 43, 40, 44, 45, 46, 47, 48, 49, 50, 51, 44, 52, 53, 54, 55, 5, 56, 3, 57, 5, 58, 59]
        token_indices = self.get_token_indices(tokens)

        character_indices = self.get_character_indices(tokens)
        # spans = [(mention['token_begin'], mention['token_end']) for mention in data['mentions']]
        text_embedder = {
            'tokens': torch.LongTensor(token_indices),
            'characters': character_indices,
            'whitespace': self.get_whitespace_indices(None),
            # token-indices are NOT 'tokens', 'tokens' are token ids as appear in dictionary, 'token-indices' are the
            # indices of tokens inside embeddings for a particular batch. So if a particular batch has 200 different words
            # there will be 200 token-indices from 0 to 199 ; whereas 'tokens' will be also 200 but ranging from 0 to
            # thousands in value
            'tokens-indices': torch.LongTensor(get_token_buckets(tokens)),
            'text': tokens
        }

        return {
            'id': None,
            'metadata_tags': ['test'],
            'xxx': text_embedder,
            'content': input_text,
            'begin': torch.IntTensor(begin),
            'end': torch.IntTensor(end),
            'spans': [],
            # 'spans': spans,
            'gold_clusters': [],
            'gold_tags_indices': [],
            'clusters': torch.IntTensor([]),
            'relations2': [],
            'num_concepts': None,
            'linker_candidates': linker_candidates,
            'linker_targets': [],
            'linker_scores': linker_scores,
            'total_cand_lengths_in_gold_mentions': [],
            'linker_gold': []
        }
