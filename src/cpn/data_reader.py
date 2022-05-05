import json
import os
import random
from collections import Counter
from time import sleep
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cpn.tokenizer import TokenizerCPN
from datass.dictionary import Dictionary
from datass.linefile import LineFileReader
from datass.transform import get_token_buckets
from modules.ner.spanner import create_all_spans
from modules.utils.misc import indices_to_spans


class TokenizerSimple:

    def tokenize(self, text):
        output = []

        offset = 0
        for token in text.split(' '):
            begin = text.find(token, offset)
            output.append({
                'offset': begin,
                'length': len(token),
                'token': token
            })
            offset = begin + len(token)

        return output


class InstanceLoader:

    def __init__(self, filename, accept, convert):
        self.file = LineFileReader(filename)
        self.convert = convert
        self.indices = []

        print("Filtering data set...")
        for idx in tqdm(range(self.file.size)):
            data = self.file.read(idx)
            data = json.loads(data.rstrip())
            if accept(data):
                convert(data)
                self.indices.append(idx)
        print("done. (", len(self.indices), "/", self.file.size, ")")

    def __getitem__(self, idx):
        data = self.file.read(self.indices[idx])
        data = json.loads(data.rstrip())
        return self.convert(data)

    def __len__(self):
        return len(self.indices)


class AbstractDataReader():
    def __init__(self):
        self.linking_candidates = None
        self.train_linker_tag = None
        self.max_link_candidates = None
        self.include_nill_in_candidates = None
        self.include_none_in_candidates = None
        self.dict_entities: Dictionary = None
        self.dict_words: Dictionary = None
        self.dict_characters: Dictionary = None
        self.dict_whitespace: Dictionary = None
        self.shuffle_candidates = None

    def get_whitespace_indices(self, whitespace):
        return torch.LongTensor(
            [self.dict_whitespace.lookup(ws) for ws in whitespace]) if whitespace is not None else None

    def get_character_indices(self, tokens):
        output = []
        for token in tokens:
            token = '<' + token + '>'
            output.append([self.dict_characters.lookup(c) for c in token])
        return output

    def get_token_indices(self, tokens):
        return [self.dict_words.lookup(token) for token in tokens]

    def get_linker_targets_all_spans(self, data, all_spans, all_spans_candidates):
        """
        """
        if self.train_linker_tag is None or self.train_linker_tag in data['tags']:
            span_to_gold = dict()
            for mention in data['mentions']:
                if is_link_trainable(mention):
                    entity_concept = data['concepts'][mention['concept']['concept']]
                    if 'link' in entity_concept:
                        mention_correct = entity_concept['link']
                        mention_span = (mention['token_begin'], mention['token_end'])
                        span_to_gold[mention_span] = mention_correct

            targets = []
            for span_idx, curr_span in enumerate(all_spans):
                # if curr_span not in span_to_gold and not include_none:
                #     index = -1
                # else:
                if curr_span not in span_to_gold:
                    correct_link = 'NONE'
                else:
                    correct_link = span_to_gold[curr_span]

                correct_link_id = self.dict_entities.add(correct_link)
                target_index = (all_spans_candidates[span_idx] == correct_link_id).nonzero()
                if target_index.size()[0] == 0:
                    index = -1
                else:
                    index = target_index.item()

                targets.append(index)
        else:
            # no linking annotation
            targets = [-1 for span in all_spans]

        return targets

    def get_linker_candidates_all_spans(self, input_content, tags, all_spans, begin, end, span_mask):
        # no linking for span: empty candidate list
        lc = self.linking_candidates
        if self.train_linker_tag is None or self.train_linker_tag in tags:
            candidates = []
            candidates_scores = []
            for idx_span, curr_span in enumerate(all_spans):
                if span_mask[idx_span] < 0.9:  # span_mask to not evaluate invalid spans (outside content boundaries)
                    candidates.append(torch.tensor([], dtype=torch.int))
                    candidates_scores.append(torch.tensor([], dtype=torch.float))
                    continue

                span_text = input_content[begin[curr_span[0]]:end[curr_span[1]]].strip()
                if span_text in lc:
                    span_candidates = lc[span_text]['candidates']
                    span_scores = lc[span_text]['scores']
                    if self.max_link_candidates is not None and self.max_link_candidates > -1:
                        span_candidates = span_candidates[:self.max_link_candidates]
                        span_scores = span_scores[:self.max_link_candidates]

                    if self.include_nill_in_candidates:
                        span_candidates = torch.cat((span_candidates,
                                                     torch.tensor([self.dict_entities.lookup('NILL')],
                                                                  dtype=torch.int32)))
                        span_scores = torch.cat((span_scores, torch.tensor([1.0])))
                    if self.include_none_in_candidates:
                        span_candidates = torch.cat((span_candidates,
                                                     torch.tensor([self.dict_entities.lookup('NONE')],
                                                                  dtype=torch.int32)))
                        span_scores = torch.cat((span_scores, torch.tensor([1.0])))

                    # candidates.append(span_candidates)
                    # candidates_scores.append(span_scores)
                else:
                    empty_candidates = []
                    empty_cand_scores = []
                    if self.include_nill_in_candidates:
                        empty_candidates.append(self.dict_entities.lookup('NILL'))
                        empty_cand_scores.append(1.0)

                    if self.include_none_in_candidates:
                        empty_candidates.append(self.dict_entities.lookup('NONE'))
                        empty_cand_scores.append(1.0)

                    span_candidates = torch.tensor(empty_candidates, dtype=torch.int32)
                    span_scores = torch.tensor(empty_cand_scores, dtype=torch.float)

                if self.shuffle_candidates:
                    # way 1: tolist and zip have to be used with this way
                    # both_cands_and_scores = list(zip(span_candidates.tolist(), span_scores.tolist()))
                    # random.shuffle(both_cands_and_scores)
                    # span_candidates, span_scores = zip(*both_cands_and_scores)
                    # span_candidates = torch.tensor(span_candidates, dtype=torch.int)
                    # span_scores = torch.tensor(span_scores, dtype=torch.float)

                    # way 2: this way seems cleaner just with indexes:
                    indices = torch.randperm(span_candidates.shape[0])
                    span_candidates = span_candidates[indices]
                    span_scores = span_scores[indices]

                candidates.append(span_candidates)
                candidates_scores.append(span_scores)

        else:
            candidates = [torch.tensor([], dtype=torch.int) for span in all_spans]
            candidates_scores = [torch.tensor([], dtype=torch.int) for span in all_spans]

        return candidates, candidates_scores


class DatasetCPN(Dataset, AbstractDataReader):

    def __init__(self, name, config, dictionaries, linking_candidates=None):
        super().__init__()
        self.name = name
        self.tokenize = config['dataset']['tokenize']
        self.tag = config['dataset']['tag']
        self.dict_words = dictionaries['words']
        self.dict_characters = dictionaries['characters']
        self.dict_whitespace = dictionaries.get('whitespace', None)
        self.dict_tags = dictionaries['tags']
        self.dict_relations = dictionaries['relations']
        self.dict_entities: Dictionary = dictionaries.get('entities', None)
        self.linking_candidates = linking_candidates
        self.max_span_length = config['model']['max_span_length']
        self.include_nill_in_candidates = config['dataloader']['include_nill_in_candidates']
        self.include_none_in_candidates = config['dataloader']['include_none_in_candidates']
        self.all_spans_candidates = config['dataloader']['all_spans_candidates']
        self.doc_level_candidates = config['dataloader']['doc_level_candidates']
        self.candidates_from_dictionary = config['dataloader']['candidates_from_dictionary']

        # TODO: if end to end: here already loads (lowercase! strip!) the dictionary
        self.shuffle_candidates = config['dataset']['shuffle_candidates']

        self.max_link_candidates = config['dataset'].get('max_link_candidates', None)

        self.train_linker_tag = config['dataset'].get('train_linker_tag', 'annotation::links')

        if self.tokenize:
            self.tokenizer = TokenizerCPN()
        path = config['dataset']['filename']

        self.instances = []
        self.number_of_lost_mentions = 0

        if config['dataset']['load-in-memory']:
            print('NR OF OOV AND TOKENS IN WORDS DITIONARY data_reader BEFORE LOADING: ')
            print('\t OOV: ', self.dict_words.out_of_voc)
            print('\t NR OF TOKENS: ', len(self.dict_words.word2idx))

            print("Loading {} tokenize={} tag={}".format(path, self.tokenize, self.tag))

            if os.path.isdir(path):
                for filename in tqdm(os.listdir(path)):
                    f = os.path.join(path, filename)
                    self.load_file(f)

            else:
                self.load_file(path)

            print("done.")
        else:
            _accept = lambda data: self.tag in data['tags']
            _convert = lambda data: self.convert(data)
            self.instances = InstanceLoader(path, _accept, _convert)

        print('NR OF OOV AND TOKENS IN WORDS DITIONARY data_reader AFTER LOADING: ')
        print('\t OOV: ', self.dict_words.out_of_voc)
        print('\t NR OF TOKENS: ', len(self.dict_words.word2idx))
        print("Number of instances in {}: {}.".format(self.name, len(self)))
        print("Number of mentions lost due to tokenization: {}".format(self.number_of_lost_mentions))
        print("Shuffle candidates:", self.shuffle_candidates)
        self.print_histogram_of_span_length()

    def print_histogram_of_span_length(self):
        counter = Counter()
        total = 0
        fail = 0
        for instance in self.instances:
            for begin, end in instance['spans']:
                if begin is None or end is None:
                    fail += 1
                else:
                    counter[end - begin] += 1
                    total += 1

        print("span\tcount\trecall")
        cum = 0
        for span_length in sorted(counter.keys()):
            count = counter[span_length]
            cum += count
            print("{}\t{}\t{}".format(span_length, count, cum / total))
        print()
        print("failed spans:", fail)

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

    def get_top_candidates(self, mention_candidates: List, mention_candidate_scores: List):
        if self.max_link_candidates is not None and self.max_link_candidates > 0:
            if len(mention_candidates) > 0:
                candidates_with_scores = zip(mention_candidates, mention_candidate_scores)
                candidates_with_scores = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)
                candidates_with_scores = candidates_with_scores[:self.max_link_candidates]
                mention_candidates = [cand[0] for cand in candidates_with_scores]
                mention_candidate_scores = [cand[1] for cand in candidates_with_scores]
        return mention_candidates, mention_candidate_scores

    def convert(self, data):
        identifier = data['id']
        # print('(kzaporoj) - currently I am converting ', identifier)
        # mentions = data['mentions']
        concepts = data['concepts']

        linker_candidates = None
        linker_targets = None
        linker_scores = None

        if self.tokenize:
            tokens = self.tokenizer.tokenize(data['content'])
            begin = [token['offset'] for token in tokens]
            end = [token['offset'] + token['length'] for token in tokens]
            tokens = [token['token'] for token in tokens]
            whitespace = None
        else:
            tokens = data['tokenization']['tokens']
            begin = data['tokenization']['begin']
            end = data['tokenization']['end']
            whitespace = data['tokenization'].get('ws', None)

        n_tokens = len(tokens)
        if n_tokens == 0:
            print("WARNING: dropping empty document")
            return

        begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
        end_to_index = {pos: idx for idx, pos in enumerate(end)}

        # this makes life easier --> assigns 'NILL' to concepts where the link is in None.
        for concept in concepts:
            concept['mentions'] = []
            if 'link' in concept and concept['link'] is None:
                concept['link'] = 'NILL'

        # 19/02/2021 - we ignore the mentions whose offsets do not match with the tokenized content (tokens)
        len_prev_mentions = len(data['mentions'])
        data['mentions'] = [men for men in data['mentions'] if men['begin'] in begin_to_index and
                            men['end'] in end_to_index]
        len_after_mentions = len(data['mentions'])
        if len_prev_mentions != len_after_mentions:
            # print('NR of lost mentions due to tokenization: ', len_prev_mentions - len_after_mentions)
            assert len_prev_mentions > len_after_mentions
            self.number_of_lost_mentions += (len_prev_mentions - len_after_mentions)

        mentions = data['mentions']

        # only candidates for each of the gold mentions
        all_doc_candidates_no_nill = list()
        for mention in mentions:
            if mention['concept'] >= len(concepts):
                raise BaseException("invalid mention concept", mention['concept'], "in doc", identifier)
            concept = concepts[mention['concept']]

            mention_candidates = []
            mention_candidate_scores = []
            mention_link = None
            if self.candidates_from_dictionary:
                if self.linking_candidates is not None and mention['text'].strip() in self.linking_candidates:
                    mention_candidates = [self.dict_entities.get(tok_ent) for tok_ent in
                                          self.linking_candidates[mention['text'].strip()]['candidates'].tolist()]
                    mention_candidate_scores = [curr_score for curr_score in
                                                self.linking_candidates[mention['text'].strip()]['scores'].tolist()]
            else:
                if 'candidates' in mention:
                    mention_candidates = mention['candidates']
                    mention_candidate_scores = mention['scores']

            mention_candidates, mention_candidate_scores = self.get_top_candidates(mention_candidates,
                                                                                   mention_candidate_scores)

            if 'link' in concept:
                mention['link'] = None
                if concept['link'] is not None:
                    curr_candidates = set()
                    if 'candidates' in mention:
                        curr_candidates = set(mention_candidates)
                    else:
                        pass
                        # print('WARNING {} - no candidates for mention ({},{}) - {}'.format(identifier,
                        #                                                                    mention['begin'],
                        #                                                                    mention['end'],
                        #                                                                    mention['text']))

                    if concept['link'] in curr_candidates:
                        mention['link'] = concept['link']

            mention['concept'] = concept
            mention['token_begin'] = begin_to_index.get(mention['begin'], None)
            mention['token_end'] = end_to_index.get(mention['end'], None)
            mention['candidates'] = mention_candidates
            mention['scores'] = mention_candidate_scores
            if mention['token_begin'] is None or mention['token_end'] is None:
                raise Exception('token_begin and token_end should not be None')
                # self.number_of_lost_mentions += 1
                # if mention['token_begin'] is None:
                #     print('token begin wrong in ', identifier, ' -- "', mention['text'], '" (',
                #           mention['begin'], '-', mention['end'], ') ... ADAPTING... ')
                #     for offset in range(len(mention['text']) * 2):
                #         if mention['begin'] - offset in begin_to_index:
                #             mention['token_begin'] = begin_to_index.get(mention['begin'] - offset)
                #             break
                # if mention['token_end'] is None:
                #     print('token end wrong in ', identifier, ' -- "', mention['text'], '" (',
                #           mention['begin'], '-', mention['end'], ') ... ADAPTING... ')
                #     for offset in range(len(mention['text']) * 2):
                #         if mention['end'] + offset in end_to_index:
                #             mention['token_end'] = end_to_index.get(mention['end'] + offset)
                #             break
                #
                # if mention['token_end'] is not None and mention['token_begin'] is not None:
                #     print('ADAPTED TO: ', tokens[mention['token_begin']:mention['token_end'] + 1])
                # else:
                #     print('COULDN\'T ADAPT!!!')

            concept['mentions'].append(mention)

            if 'candidates' in mention:
                mention['candidates'] = [x if x is not None else 'NILL' for x in mention['candidates']]
                if 'NILL' not in mention['candidates']:
                    mention['candidates'].append('NILL')
                    mention['scores'].append(1.0)

                # the length of all the candidates (including NILL)
                mention['total_cand_lengths_in_gold_mentions'] = len(mention['candidates'])
                if not self.include_nill_in_candidates:
                    mention['candidates'] = [cand for cand in mention['candidates'] if cand != 'NILL']
                if self.shuffle_candidates:
                    random.shuffle(mention['candidates'])
                if self.doc_level_candidates:
                    all_doc_candidates_no_nill.extend(mention['candidates'])

            if 'link' in mention and mention['link'] is None:
                mention['link'] = 'NILL'

        if self.doc_level_candidates:
            all_doc_candidates_no_nill = list(set(all_doc_candidates_no_nill))
            if self.shuffle_candidates:
                random.shuffle(all_doc_candidates_no_nill)
            data['all_doc_candidates_no_nill'] = all_doc_candidates_no_nill

        # TODO - begin end-to-end span loading
        all_possible_spans = []
        linker_total_length_cands = []
        if self.all_spans_candidates:
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
                self.get_linker_candidates_all_spans(data['content'], data['tags'],
                                                     all_spans=all_possible_spans, begin=begin, end=end,
                                                     span_mask=span_masked_scores[0])

            linker_targets_all_spans = self.get_linker_targets_all_spans(data, all_possible_spans,
                                                                         linker_cands_all_spans)

            linker_candidates = linker_cands_all_spans
            linker_targets = linker_targets_all_spans
            linker_scores = linker_cands_all_spans_scores
        else:
            # linker candidates
            linker_total_length_cands, linker_candidates = self.get_candidates_from_mentions(data)
            linker_targets = self.get_targets_from_mentions(data, all_doc_candidates=self.doc_level_candidates,
                                                            is_nill_in_candidates=self.include_nill_in_candidates)

        linker_gold = self.get_linker_gold_from_mentions(data)

        # TODO - end end-to-end span loading

        token_indices = self.get_token_indices(tokens)

        character_indices = self.get_character_indices(tokens)
        spans = [(mention['token_begin'], mention['token_end']) for mention in data['mentions']
                 # 19/02/2021 - adding this because in dwie public the tokenization sometimes does not match
                 # if mention['token_begin'] is not None and mention['token_end'] is not None
                 ]

        gold_clusters = [[(mention['token_begin'], mention['token_end']) for mention in concept['mentions']
                          # 19/02/2021 - adding this because in dwie public the tokenization sometimes does not match
                          # if mention['token_begin'] is not None and mention['token_end'] is not None
                          ] for concept
                         in concepts]

        # can be the case that mentions are filtered with either token_begin or token_end in None (see above)
        gold_clusters = [concept for concept in gold_clusters if len(concept) > 0]

        text_embedder = {
            'tokens': torch.LongTensor(token_indices),
            'characters': character_indices,
            'whitespace': self.get_whitespace_indices(whitespace),
            'tokens-indices': torch.LongTensor(get_token_buckets(tokens)),
            'text': tokens
        }

        # TODO: rename variables to be more clear
        return {
            'id': identifier,
            'metadata_tags': data['tags'],
            'xxx': text_embedder,
            'content': data['content'],
            'begin': torch.IntTensor(begin),
            'end': torch.IntTensor(end),
            'spans': spans,
            # 'all_possible_spans': all_possible_spans,
            'gold_clusters': gold_clusters,
            'gold_tags_indices': self.get_span_tags(mentions),
            'clusters': torch.IntTensor([mention['concept']['concept'] for mention in mentions
                                         # 19/02/2021 - adding this because in dwie public the tokenization sometimes does not match
                                         # if mention['token_begin'] is not None and mention['token_end'] is not None
                                         ]),
            'relations2': self.get_relations(data),
            # 'num_concepts': len(concepts),
            # 19/02/2021 - changed this one because can be different if no tokenization found for mentions (see above)
            'num_concepts': len(gold_clusters),
            # 'linker_candidates_no_nill_doc': linker_all_candidates,
            'linker_candidates': linker_candidates,
            'linker_targets': linker_targets,
            'linker_scores': linker_scores,
            'total_cand_lengths_in_gold_mentions': linker_total_length_cands,
            # 'linker_targets_no_nill_doc': linker_targets_all_no_nill,
            # 'linker_candidates_no_nill': linker_candidates_no_nill,
            # 'linker_targets_no_nill': linker_targets_no_nill,
            'linker_gold': linker_gold
            # 'linker_cands_all_spans_no_nill': linker_cands_all_spans_no_nill,
            # 'linker_cands_all_spans_no_nill_scores': linker_cands_all_spans_no_nill_scores,
            # 'linker_targets_all_spans_no_nill': linker_targets_all_spans_no_nill,
            # 'linker_cands_all_spans': linker_cands_all_spans,
            # 'linker_cands_all_spans_scores': linker_cands_all_spans_scores,
            # 'linker_targets_all_spans': linker_targets_all_spans
        }

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    # def get_token_indices(self, tokens):
    #     return [self.dict_words.lookup(token) for token in tokens]

    # def get_character_indices(self, tokens):
    #     output = []
    #     for token in tokens:
    #         token = '<' + token + '>'
    #         output.append([self.dict_characters.lookup(c) for c in token])
    #     return output

    # def get_whitespace_indices(self, whitespace):
    #     return torch.LongTensor(
    #         [self.dict_whitespace.lookup(ws) for ws in whitespace]) if whitespace is not None else None

    def get_span_tags(self, mentions):
        spans = []
        for mention in mentions:
            # if mention['token_begin'] is not None and mention['token_end'] is not None:
            assert mention['token_begin'] is not None
            assert mention['token_end'] is not None
            spans.extend([(mention['token_begin'], mention['token_end'], self.dict_tags.lookup(tag)) for tag in
                          mention['concept']['tags']])
        return spans

    def get_relations(self, data):
        return [(relation['s'], relation['o'], self.dict_relations.lookup(relation['p'])) for relation in
                data['relations']]

    def get_candidates_from_mentions(self, data):
        # no linking for span: empty candidate list

        if self.train_linker_tag in data['tags']:
            candidates = []
            total_length_cand = []

            for mention in data['mentions']:
                if is_link_trainable(mention):
                    curr_cand = [self.dict_entities.add(c) for c in mention['candidates']]
                    candidates.append(torch.tensor(curr_cand, dtype=torch.int32))
                    total_length_cand.append(mention['total_cand_lengths_in_gold_mentions'])
                else:
                    candidates.append(torch.tensor([], dtype=torch.int32))
                    total_length_cand.append(0)
        else:
            candidates = [torch.tensor([], dtype=torch.int32) for mention in data['mentions']]
            total_length_cand = [0 for mention in data['mentions']]

        return total_length_cand, candidates

    def get_linker_all_candidates(self, data, all_candidates):
        # no linking for span: empty candidate list

        if self.train_linker_tag in data['tags']:
            candidates = []

            candidates.extend([self.dict_entities.add(c) for c in all_candidates])
        else:
            candidates = []

        return candidates

    def get_targets_from_mentions(self, data, all_doc_candidates=False, is_nill_in_candidates=False):
        """

        :param data:
        :param type_target: 'mention_not_nill', 'mention_nill', 'all_candidates'
        :return:
        """
        if self.train_linker_tag in data['tags']:
            targets = []

            for mention in data['mentions']:
                # 19/02/2021 - adding this because in dwie public the tokenization sometimes does not match
                # if mention['token_begin'] is None or mention['token_end'] is None:
                #     continue
                assert mention['token_begin'] is not None
                assert mention['token_end'] is not None

                if is_link_trainable(mention):
                    # kzaporoj - uncommented this one , the mention['link'] doesn't seem to work: no 'link'
                    # inside mention
                    if not all_doc_candidates:
                        mention_candidates = mention['candidates']
                    else:
                        mention_candidates = data['all_doc_candidates_no_nill']

                    if mention['link'] in mention_candidates:
                        index = mention_candidates.index(
                            mention['link'])  # kzaporoj: commented this one and uncommented previous one
                    else:
                        if 'NILL' in mention_candidates:
                            index = mention_candidates.index('NILL')
                        else:
                            if (not is_nill_in_candidates) or (all_doc_candidates):
                                index = -1
                            else:
                                RuntimeError(
                                    '!!!Should not happen that NILL is not in mention_candidates and '
                                    'all_doc_candidates: ', all_doc_candidates,
                                    ' and is_nill_in_candidates: ', is_nill_in_candidates, '!!!')
                else:
                    index = -1

                targets.append(index)
        else:
            # no linking annotation
            targets = [-1 for mention in data['mentions']]

        return targets



    def get_linker_gold_from_mentions(self, data):
        gold = []
        for mention in data['mentions']:
            # 19/02/2021 - sometimes if tokenization is not good, some mentions should be ignored (because there is no
            # token that can be assigned to either the begin or end of the mention)
            # if mention['token_begin'] is None or mention['token_end'] is None:
            #     continue
            assert mention['token_begin'] is not None
            assert mention['token_end'] is not None

            concept = mention['concept']

            if 'link' in concept:
                gold.append((mention['token_begin'], mention['token_end'], concept['link']))

            # (kzaporoj) - the next gold calculation is wrong because only adds correct link if it is present in candidate
            # list. It doesn't account for the fact that the candidate list can be wrong. Coreflinker can take the link
            # from other mentions in the same cluster, giving lower results when using this formulation
            # while the output is actually correct.

            # if 'link' in mention:
            #     gold.append((mention['token_begin'], mention['token_end'], mention['link']))
        return gold


def is_link_trainable(mention):
    # if mention['token_begin'] is None or mention['token_end'] is None:
    #     return False
    assert mention['token_begin'] is not None
    assert mention['token_end'] is not None

    return 'candidates' in mention and 'link' in mention  # kzaporoj - commented this, can not find 'link' in mention field
    # return 'candidates' in mention  # kzaporoj - instead have put this one



################################## OTHER COMMENTED STUFF
    # def get_linker_candidates_all_spans(self, data, all_spans, begin, end, span_mask):
    #     # no linking for span: empty candidate list
    #     lc = self.linking_candidates
    #     if self.train_linker_tag in data['tags']:
    #         candidates = []
    #         candidates_scores = []
    #         for idx_span, curr_span in enumerate(all_spans):
    #             if span_mask[idx_span] < 0.9:  # span_mask to not evaluate invalid spans (outside content boundaries)
    #                 candidates.append(torch.tensor([], dtype=torch.int))
    #                 candidates_scores.append(torch.tensor([], dtype=torch.float))
    #                 continue
    #
    #             span_text = data['content'][begin[curr_span[0]]:end[curr_span[1]]].strip()
    #             if span_text in lc:
    #                 span_candidates = lc[span_text]['candidates']
    #                 span_scores = lc[span_text]['scores']
    #                 if self.max_link_candidates is not None and self.max_link_candidates > -1:
    #                     span_candidates = span_candidates[:self.max_link_candidates]
    #                     span_scores = span_scores[:self.max_link_candidates]
    #
    #                 if self.include_nill_in_candidates:
    #                     span_candidates = torch.cat((span_candidates,
    #                                                  torch.tensor([self.dict_entities.lookup('NILL')],
    #                                                               dtype=torch.int32)))
    #                     span_scores = torch.cat((span_scores, torch.tensor([1.0])))
    #                 if self.include_none_in_candidates:
    #                     span_candidates = torch.cat((span_candidates,
    #                                                  torch.tensor([self.dict_entities.lookup('NONE')],
    #                                                               dtype=torch.int32)))
    #                     span_scores = torch.cat((span_scores, torch.tensor([1.0])))
    #
    #                 # candidates.append(span_candidates)
    #                 # candidates_scores.append(span_scores)
    #             else:
    #                 empty_candidates = []
    #                 empty_cand_scores = []
    #                 if self.include_nill_in_candidates:
    #                     empty_candidates.append(self.dict_entities.lookup('NILL'))
    #                     empty_cand_scores.append(1.0)
    #
    #                 if self.include_none_in_candidates:
    #                     empty_candidates.append(self.dict_entities.lookup('NONE'))
    #                     empty_cand_scores.append(1.0)
    #
    #                 span_candidates = torch.tensor(empty_candidates, dtype=torch.int32)
    #                 span_scores = torch.tensor(empty_cand_scores, dtype=torch.float)
    #
    #             if self.shuffle_candidates:
    #                 # way 1: tolist and zip have to be used with this way
    #                 # both_cands_and_scores = list(zip(span_candidates.tolist(), span_scores.tolist()))
    #                 # random.shuffle(both_cands_and_scores)
    #                 # span_candidates, span_scores = zip(*both_cands_and_scores)
    #                 # span_candidates = torch.tensor(span_candidates, dtype=torch.int)
    #                 # span_scores = torch.tensor(span_scores, dtype=torch.float)
    #
    #                 # way 2: this way seems cleaner just with indexes:
    #                 indices = torch.randperm(span_candidates.shape[0])
    #                 span_candidates = span_candidates[indices]
    #                 span_scores = span_scores[indices]
    #
    #             candidates.append(span_candidates)
    #             candidates_scores.append(span_scores)
    #
    #     else:
    #         candidates = [torch.tensor([], dtype=torch.int) for span in all_spans]
    #         candidates_scores = [torch.tensor([], dtype=torch.int) for span in all_spans]
    #
    #     return candidates, candidates_scores



    # def get_linker_targets_all_spans(self, data, all_spans, all_spans_candidates):
    #     """
    #     """
    #     if self.train_linker_tag in data['tags']:
    #         span_to_gold = dict()
    #         for mention in data['mentions']:
    #             if is_link_trainable(mention):
    #                 entity_concept = data['concepts'][mention['concept']['concept']]
    #                 # if 'link' in entity_concept and entity_concept['link'] is not None \
    #                 #         and entity_concept['link'] != 'NILL':
    #                 if 'link' in entity_concept:
    #                     mention_correct = entity_concept['link']
    #                     mention_span = (mention['token_begin'], mention['token_end'])
    #                     span_to_gold[mention_span] = mention_correct
    #             else:
    #                 RuntimeError('!!!For some reason the mention is not linkable ', mention, '!!!')
    #
    #         targets = []
    #         for span_idx, curr_span in enumerate(all_spans):
    #             # if curr_span not in span_to_gold and not include_none:
    #             #     index = -1
    #             # else:
    #             if curr_span not in span_to_gold:
    #                 correct_link = 'NONE'
    #             else:
    #                 correct_link = span_to_gold[curr_span]
    #
    #             correct_link_id = self.dict_entities.add(correct_link)
    #             target_index = (all_spans_candidates[span_idx] == correct_link_id).nonzero()
    #             if target_index.size()[0] == 0:
    #                 index = -1
    #             else:
    #                 index = target_index.item()
    #
    #             targets.append(index)
    #     else:
    #         # no linking annotation
    #         targets = [-1 for span in all_spans]
    #
    #     return targets