# (kzaporoj 09/04/2021) - This is a very similar to data_reader_bert.py, BUT the main idea is to produce as much
# output in tensor format as possible; avoid the use of lists that later on get converted to tensors inside of
# the collate_dygie_spanbert function when the model is training. The reason is that having tensors already should
# make it faster.
import json
import os
import random
from time import sleep

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datass.dictionary import Dictionary


class DatasetDWIESpanBertHoi(Dataset):
    def __init__(self, name, config, dictionaries, linking_candidates=None):
        # print('TODO FOR ')
        super().__init__()
        self.name = name
        self.tag = config['dataset']['tag']
        self.bert_max_segment_len = config['dataloader']['bert_max_segment_len']
        self.instances = []
        self.dict_tags = dictionaries['tags']

        self.include_nill_in_candidates = config['dataloader']['include_nill_in_candidates']
        self.include_none_in_candidates = config['dataloader']['include_none_in_candidates']

        self.bert_dictionary = dictionaries['bert_subtokens']
        self.dict_relations = dictionaries['relations']
        self.dict_entities: Dictionary = None
        if 'entities' in dictionaries:
            self.dict_entities: Dictionary = dictionaries['entities']

        self.max_span_length = config['model']['max_span_length']  # TODO

        self.is_baseline_linker = False
        if config['output_config']['output_content']:
            self.output_content = True
        else:
            self.output_content = False
        if 'linker' in config['model']:
            self.is_baseline_linker = config['model']['linker']['enabled']

        # print('NR OF OOV AND TOKENS IN WORDS DITIONARY data_reader BEFORE LOADING: ')
        # print('\t OOV: ', self.dict_words.out_of_voc)
        # print('\t NR OF TOKENS: ', len(self.dict_words.word2idx))

        path = config['dataset']['filename']

        print("Loading {} tag={}".format(path, self.tag))

        if os.path.isdir(path):
            for filename in tqdm(os.listdir(path)):
                f = os.path.join(path, filename)
                self.load_file(f)

        else:
            self.load_file(path)

        print("done init in DatasetDWIESpanBert")

    def get_linker_gold_from_mentions(self, data):
        gold = []
        for mention in data['mentions']:
            # 19/02/2021 - sometimes if tokenization is not good, some mentions should be ignored (because there is no
            # token that can be assigned to either the begin or end of the mention)
            # if mention['token_begin'] is None or mention['token_end'] is None:
            #     continue
            assert mention['subtoken_begin'] is not None
            assert mention['subtoken_end'] is not None

            concept = data['concepts'][mention['concept']]

            if 'link' in concept:
                gold.append((mention['subtoken_begin'], mention['subtoken_end'], concept['link']))

            # (kzaporoj) - the next gold calculation is wrong because only adds correct link if it is present in candidate
            # list. It doesn't account for the fact that the candidate list can be wrong. Coreflinker can take the link
            # from other mentions in the same cluster, giving lower results when using this formulation
            # while the output is actually correct.

            # if 'link' in mention:
            #     gold.append((mention['token_begin'], mention['token_end'], mention['link']))
        return gold

    def convert(self, data):
        identifier = data['id']
        concepts = data['concepts']

        # max_span_length = self

        subtokens = data['bert_tokenization']
        # subtoken_map = torch.tensor(data['subtoken_map'], dtype=torch.int)
        # token_length = subtoken_map.max().item() + 1  # + 1 because zero-based

        subtoken_map = data['subtoken_map']
        token_length = max(subtoken_map) + 1  # + 1 because zero-based
        bert_segments = data['bert_segments']
        # ['tokens']
        # begin = data['bert_tokenization']['begin']
        # end = data['bert_tokenization']['end']
        # print('DatasetDWIESpanBert: converting for ', identifier)

        mentions = data['mentions']

        for idx_mention, curr_mention in enumerate(mentions):
            curr_mention['candidates'] = []  # TODO
            curr_mention['scores'] = []  # TODO
            men_concept = data['concepts'][curr_mention['concept']]
            if 'mentions' not in men_concept:
                men_concept['mentions'] = []
            men_concept['mentions'].append(curr_mention)

        spans = [(mention['subtoken_begin'], mention['subtoken_end']) for mention in data['mentions']]
        gold_clusters = [[(mention['subtoken_begin'], mention['subtoken_end']) for mention in concept['mentions']]
                         for concept in concepts if 'mentions' in concept]
        gold_clusters = [concept for concept in gold_clusters if len(concept) > 0]

        linker_candidates = [torch.IntTensor(cnd) for cnd in data['all_spans_candidates']]
        linker_scores = [torch.FloatTensor(cnd) for cnd in data['all_spans_candidates_scores']]
        linker_targets = data['all_spans_candidates_target']

        bert_segms, bert_segs_mask = [], []

        num_subtokens = sum([len(s) for s in bert_segments])

        # creates segments
        for idx, sent_tokens in enumerate(bert_segments):
            curr_subtoken_ids = self.bert_dictionary.convert_tokens_to_ids(sent_tokens)
            curr_subtoken_mask = [1] * len(curr_subtoken_ids)
            while len(curr_subtoken_ids) < self.bert_max_segment_len:
                curr_subtoken_ids.append(0)
                curr_subtoken_mask.append(0)
            bert_segms.append(curr_subtoken_ids)
            bert_segs_mask.append(curr_subtoken_mask)

        # bert_segments = np.array(bert_segments)
        # bert_segs_mask = np.array(bert_segs_mask)

        bert_segms = torch.tensor(bert_segms, dtype=torch.long)
        bert_segs_mask = torch.tensor(bert_segs_mask, dtype=torch.long)

        assert num_subtokens == bert_segs_mask.sum(), (num_subtokens, bert_segs_mask.sum())

        # filters spans by width
        f_all_spans = list()
        f_linker_candidates = list()
        f_linker_targets = list()
        f_linker_scores = list()
        for curr_span, curr_candidates, curr_target, curr_scores in zip(data['all_spans'], linker_candidates,
                                                                        linker_targets, linker_scores):
            if curr_span[1] - curr_span[0] < self.max_span_length:
                f_all_spans.append(curr_span)
                if not self.include_nill_in_candidates:
                    curr_candidates = curr_candidates[:-1]
                    curr_scores = curr_scores[:-1]
                    if curr_target >= curr_candidates.shape[0]:
                        curr_target = -1

                f_linker_candidates.append(curr_candidates)
                f_linker_scores.append(curr_scores)
                if self.dict_entities is None:
                    f_linker_targets.append(-1)
                elif curr_target > -1 and curr_candidates[curr_target].item() == \
                        self.dict_entities.lookup('NILL') and not self.include_nill_in_candidates:
                    f_linker_targets.append(-1)
                elif curr_target > -1 and curr_candidates[curr_target].item() == \
                        self.dict_entities.lookup('NONE') and not self.include_none_in_candidates:
                    f_linker_targets.append(-1)
                elif self.include_nill_in_candidates and curr_target == -1:
                    # link does not exist in candidate list, if we are running the baseline, this would be equivalent
                    # of nill and belongs to unsolvable cases by the baseline, since it strictly depends on the
                    # elements in the candidate list; the prediction of this NILL will result in a false negative
                    # impacting the recall (accuracy)

                    nill_id = self.dict_entities.lookup('NILL')
                    nill_idx = (curr_candidates == nill_id).nonzero(as_tuple=True)[0].item()
                    f_linker_targets.append(nill_idx)

                else:
                    f_linker_targets.append(curr_target)

        assert len(f_all_spans) == len(f_linker_candidates)
        assert len(f_all_spans) == len(f_linker_targets)
        assert len(f_all_spans) == len(f_linker_scores)

        linker_gold = self.get_linker_gold_from_mentions(data)

        # text_embedder = {
        #     'subtokens': torch.LongTensor(self.bert_dictionary.convert_tokens_to_ids(subtokens)),
        # }

        to_ret = {
            'id': identifier,
            'metadata_tags': data['tags'],
            # 'gold_tags_indices': self.get_span_tags(mentions, concepts),
            # 'content': data['content'], # TODO: do we need this??? should take considerable space in memory I guess
            'bert_segments': bert_segms,
            'bert_segs_mask': bert_segs_mask,
            # 'relations2': self.get_relations(data),
            'num_concepts': len(gold_clusters),
            'gold_clusters': gold_clusters,
            # TODO: check this comparing with data_reader.py where mention['concept']['concept'] is returned
            'clusters': torch.IntTensor([mention['concept'] for mention in mentions]),
            # 'spans': spans,
            # TODO: we are here incorporating this information in 'gold_starts' of pruner.py
            'gold_subtokens_start': torch.IntTensor([s[0] for s in spans]),
            'gold_subtokens_end': torch.IntTensor([s[1] for s in spans]),
            # 'all_spans': f_all_spans,
            'all_spans_tensor': torch.tensor(f_all_spans),
            'linker_candidates': f_linker_candidates,
            'linker_targets': f_linker_targets,
            'linker_scores': f_linker_scores,
            'begin_token': data['begin_token'],
            'end_token': data['end_token'],
            # 'all_spans': data['all_spans'],
            # 'linker_candidates': linker_candidates,
            # 'linker_targets': linker_targets,
            # 'linker_scores': linker_scores,
            'linker_gold': linker_gold,
            'subtoken_map': subtoken_map,
            'token_length': token_length,
            'sentence_map': torch.IntTensor(data['sentence_map'])
        }

        # TODO: transform to torch
        if self.output_content:
            to_ret['content'] = data['content']
        else:
            to_ret['content'] = ''

        return to_ret

    def load_file(self, filename, retry=0):
        if filename.endswith('.json'):
            try:
                self.load_json(filename)
            except OSError as exept:
                # tries 3 more times with random sleep
                if retry < 10:
                    print('following exept: ', exept.strerror)
                    print('except to load, trying again: ', filename, ' for retry: ', retry)
                    sleep(random.randint(5, 10))
                    self.load_file(filename, retry=retry + 1)
                else:
                    print('NO MORE TERIES LEFT, FAILING')
                    raise exept
        elif filename.endswith('.jsonl'):
            self.load_jsonl(filename)
        else:
            raise BaseException("unknown file type:", filename)

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            if 'tags' in data:
                if self.tag in data['tags']:
                    self.instances.append(self.convert(data))
            else:
                print('WARNING (kzaporoj) - NO tags IN ', filename)

    def load_jsonl(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                data = json.loads(line.rstrip())
                if self.tag in data['tags']:
                    self.instances.append(self.convert(data))

    def get_span_tags(self, mentions, concepts):
        spans = []
        for mention in mentions:
            # if mention['token_begin'] is not None and mention['token_end'] is not None:
            assert mention['subtoken_begin'] is not None
            assert mention['subtoken_end'] is not None
            spans.extend([(mention['subtoken_begin'], mention['subtoken_end'], self.dict_tags.lookup(tag)) for tag in
                          concepts[mention['concept']]['tags']])
        return spans

    def get_relations(self, data):
        return [(relation['s'], relation['o'], self.dict_relations.lookup(relation['p'])) for relation in
                data['relations']]

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)
