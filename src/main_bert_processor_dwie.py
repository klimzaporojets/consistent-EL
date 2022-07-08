""" The goal of this module is to take as input the original DW_x.json files and to tokenize them using
SpanBert tokenizer and map the original tokens to the bert-tokenized. Also this module will segment the
file in bert subtokens. This same pre-processing procedure is used by other state-of-the-art works such as
https://github.com/lxucs/coref-hoi/"""
import argparse
import json
import logging
import os
import pickle
from typing import Dict, List

import torch
from transformers import BertTokenizer

from data_processing.bert_preprocessing import normalize_word, flatten, get_sentence_map
from data_processing.dictionary import Dictionary
from models.misc.spanner import create_all_spans
from models.utils.misc import indices_to_spans

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonBertDocument(object):

    def __init__(self, json_doc, args):
        self.doc_key = json_doc['id']
        self.content = json_doc['content']
        self.word_tokenization = json_doc['tokenization']
        self.tags = json_doc['tags']
        self.concepts = json_doc['concepts']
        self.relations = json_doc['relations']

        self.tokens = []

        # begin and end of tokens (NOT BERT subtokens, just word tokens)
        self.begin_token = []
        self.end_token = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        self.subtokens_begin = []
        self.subtokens_end = []

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []
        self.mentions = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []

        # self.spans_to_candidate_links = dict()  # all the spans pointing to candidates
        self.all_spans = list()  # all the spans pointing to candidates
        self.all_spans_candidates = list()  # candidates for the spans
        self.all_spans_candidates_scores = list()  # the scores for each of the candidates
        self.all_spans_candidates_targets = list()  # the target ids of the candidate list for each of the spans

    def convert_to_json(self):
        """ Returns the JSON in DWIE format but with BERT sub-tokenization """

        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        num_all_seg_tokens = len(flatten(self.segments))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        #
        return {
            'id': self.doc_key,
            'content': self.content,
            'tags': self.tags,
            'begin_token': self.begin_token,
            'end_token': self.end_token,
            'word_tokenization': self.tokens,
            'bert_tokenization': self.subtokens,
            'bert_segments': self.segments,
            'mentions': self.mentions,
            'concepts': self.concepts,
            'relations': self.relations,
            'sentence_map': sentence_map,
            'subtoken_map': subtoken_map,
            'all_spans': self.all_spans,
            'all_spans_candidates': self.all_spans_candidates,
            'all_spans_candidates_scores': self.all_spans_candidates_scores,
            'all_spans_candidates_target': self.all_spans_candidates_targets
        }


def get_tokenizer(bert_tokenizer_name):
    return BertTokenizer.from_pretrained(bert_tokenizer_name)


def load_linking_candidates(cands_path, max_cands, links_dictionary: Dictionary):
    links_dictionary.add('NILL')

    candidates_path = cands_path
    max_link_candidates = max_cands
    span_text_to_candidates = dict()
    for curr_line in open(candidates_path):
        curr_span_candidates = json.loads(curr_line)
        span_text = curr_span_candidates['text'].strip()  # TODO: makes sense lowercasing, or will make it worse???
        span_candidates = curr_span_candidates['candidates']
        span_scores = curr_span_candidates['scores']
        # candidates should come sorted by score, but just in case sorts again
        sorted_candidates = sorted(zip(span_candidates, span_scores), key=lambda x: x[1], reverse=True)

        if max_link_candidates > -1:
            sorted_candidates = sorted_candidates[:max_link_candidates]

        span_text_to_candidates[span_text] = dict()

        scores_list = list()
        candidates_list = list()
        for curr_candidate, curr_score in sorted_candidates:
            candidates_list.append(links_dictionary.add(curr_candidate))
            scores_list.append(curr_score)
        # passes to torch.tensor in order to decrease the memory footprint - the lists consume too much memory in python
        # candidates_list
        span_text_to_candidates[span_text]['candidates'] = candidates_list
        span_text_to_candidates[span_text]['scores'] = scores_list

    return span_text_to_candidates, links_dictionary


class BertProcessor(object):
    def __init__(self, args, linking_candidates, links_dictionary):
        self.args = args
        self.linking_candidates: Dict = linking_candidates
        self.documents: List[Dict] = []
        self.tokenizer: BertTokenizer = None
        self.links_dictionary = links_dictionary
        self.speaker_separator = '[SPL]'

    def split_into_segments(self, bert_doc: JsonBertDocument, sentence_end, token_end):
        """ Split into segments.
            Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
            Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
        """
        curr_idx = 0  # Index for subtokens
        prev_token_idx = 0
        map_subtoks_to_segmented_subtoks = dict()
        offset = 1
        while curr_idx < len(bert_doc.subtokens):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.args.max_seg_len - 1 - 2, len(bert_doc.subtokens) - 1)  # Inclusive
            while end_idx >= curr_idx and not sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.info(f'{bert_doc.doc_key}: no sentence end found; split at token end')
                # If no sentence end point, try to split at token end point
                end_idx = min(curr_idx + self.args.max_seg_len - 1 - 2, len(bert_doc.subtokens) - 1)
                while end_idx >= curr_idx and not token_end[end_idx]:
                    end_idx -= 1
                if end_idx < curr_idx:
                    logger.error('Cannot split valid segment: no sentence end or token end')
                    raise Exception('Cannot split valid segment: no sentence end or token end')

            for i in range(curr_idx, end_idx + 1):
                map_subtoks_to_segmented_subtoks[i] = i + offset

            segment = [self.tokenizer.cls_token] + bert_doc.subtokens[curr_idx: end_idx + 1] + \
                      [self.tokenizer.sep_token]

            offset += 2
            bert_doc.segments.append(segment)

            subtoken_map = bert_doc.subtoken_map[curr_idx: end_idx + 1]
            bert_doc.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

            curr_idx = end_idx + 1
            prev_token_idx = subtoken_map[-1]
        assert len(flatten(bert_doc.segments)) == max(map_subtoks_to_segmented_subtoks.values()) + 2
        return map_subtoks_to_segmented_subtoks

    def get_linker_candidates_all_spans(self, input_content, all_spans, begin, end, span_mask):
        # no linking for span: empty candidate list
        lc = self.linking_candidates
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
                if self.args.max_nr_candidates is not None and self.args.max_nr_candidates > -1:
                    span_candidates = span_candidates[:self.args.max_nr_candidates]
                    span_scores = span_scores[:self.args.max_nr_candidates]
            else:
                span_candidates = []
                span_scores = []

            span_candidates.append(self.links_dictionary.lookup('NILL'))
            span_scores.append(1.0)

            candidates.append(span_candidates)
            candidates_scores.append(span_scores)

        return candidates, candidates_scores

    def get_linker_targets_all_spans(self, data, all_spans, all_spans_candidates):
        """
        """
        span_to_gold = dict()
        for mention in data['mentions']:
            entity_concept = data['concepts'][mention['concept']]
            if 'link' in entity_concept:
                mention_correct = entity_concept['link']
                mention_span = (mention['subtoken_begin'], mention['subtoken_end'])
                span_to_gold[mention_span] = mention_correct

        targets = []
        for span_idx, curr_span in enumerate(all_spans):
            if curr_span not in span_to_gold:
                correct_link = 'NILL'
            else:
                correct_link = span_to_gold[curr_span]

            correct_link_id = self.links_dictionary.add(correct_link)
            target_index = (all_spans_candidates[span_idx] == correct_link_id).nonzero()
            if target_index.size()[0] == 0:
                index = -1
            else:
                index = target_index.item()

            targets.append(index)

        return targets

    def get_document(self, json_doc) -> Dict:
        """ Process raw input to finalized documents """
        bert_doc: JsonBertDocument = JsonBertDocument(json_doc, args=self.args)
        word_idx = -1

        begin = json_doc['tokenization']['begin']
        end = json_doc['tokenization']['end']

        bert_doc.begin_token = begin
        bert_doc.end_token = end

        begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
        end_to_index = {pos: idx for idx, pos in enumerate(end)}
        max_end = max(end_to_index.keys())
        word_idx_to_first_subtoken_idx = dict()
        word_idx_to_last_subtoken_idx = dict()
        subtoken_idx = 0
        # Build up documents
        for idx_token, token in enumerate(json_doc['tokenization']['tokens']):
            word_idx += 1
            word = normalize_word(token)
            subtokens = self.tokenizer.tokenize(word)
            bert_doc.tokens.append(word)
            bert_doc.token_end += [False] * (len(subtokens) - 1) + [True]
            for idx, subtoken in enumerate(subtokens):
                bert_doc.subtokens.append(subtoken)
                # info = None if idx != 0 else len(subtokens)
                if idx == 0:
                    word_idx_to_first_subtoken_idx[word_idx] = subtoken_idx
                word_idx_to_last_subtoken_idx[word_idx] = subtoken_idx
                # bert_doc.info.append(info)
                if idx_token >= len(json_doc['tokenization']['tokens']) - 1 and idx >= len(subtokens) - 1:
                    # the last subtoken in the document always is the end of sentence by definition
                    bert_doc.sentence_end.append(True)
                elif token in {'.', '?', '!'}:
                    # the dot is the end of sentence as long as the next word starts with alphabetical uppercase
                    # so no decimal numbers as 99.1 are splitted in sentences.
                    if idx_token < len(json_doc['tokenization']['tokens']) - 1 and \
                            json_doc['tokenization']['tokens'][idx_token + 1][0].isupper():
                        bert_doc.sentence_end.append(True)
                    elif idx_token < len(json_doc['tokenization']['tokens']) - 2 and \
                            json_doc['tokenization']['tokens'][idx_token + 2][0].isupper() and \
                            (json_doc['tokenization']['tokens'][idx_token + 1] == '\'' or
                             json_doc['tokenization']['tokens'][idx_token + 1] == '"'):
                        bert_doc.sentence_end.append(True)
                    else:
                        bert_doc.sentence_end.append(False)
                else:
                    # in all other cases, it is not the end of sentence
                    bert_doc.sentence_end.append(False)
                bert_doc.subtoken_map.append(word_idx)
                subtoken_idx += 1

        assert len(bert_doc.subtokens) == len(bert_doc.sentence_end)
        # now maps all the mentions to the bert subtoken positions
        for mention in json_doc['mentions']:
            #### BEGIN: for debug purposes only
            if mention['end'] not in end_to_index:
                logger.warning('WARNING END, the following mention is not represented by tokens: %s' % mention)
                logger.warning('WARNING END chosing the widest one')
                # last_end = end_to_index
                while mention['end'] < max_end and mention['end'] not in end_to_index:
                    mention['end'] += 1
                if mention['end'] in end_to_index:
                    if mention['begin'] in begin_to_index:
                        logger.warning('WARNING END SUCCESS chosen the widest one: %s' %
                                       json_doc['tokenization']['tokens']
                                       [begin_to_index[mention['begin']]:end_to_index[mention['end']] + 1])
                    else:
                        logger.warning('WARNING END SUCCESS, BUT NO SHOW BECAUSE BEGIN WRONG')
                else:
                    logger.warning('WARNING END FAILURE can not do anything to adjust and of the mention, ignoring')
                    continue

            if mention['begin'] not in begin_to_index:
                logger.warning(
                    'WARNING BEGIN, the following mention is not represented by tokens (START): %s' % mention)
                logger.warning('WARNING BEGIN chosing the widest one (START)')
                # last_end = end_to_index
                while mention['begin'] > 0 and mention['begin'] not in begin_to_index:
                    mention['begin'] -= 1
                if mention['begin'] in begin_to_index:
                    logger.warning('WARNING BEGIN SUCCESS START chosen the widest one: %s' %
                                   json_doc['tokenization']['tokens']
                                   [begin_to_index[mention['begin']]:end_to_index[mention['end']] + 1])
                else:
                    logger.warning(
                        'WARNING BEGIN FAILURE START can not do anything to adjust and of the mention, ignoring')
                    continue
            #### END: for debug purposes only
            token_begin = begin_to_index[mention['begin']]
            subtoken_begin = word_idx_to_first_subtoken_idx[token_begin]
            mention['subtoken_begin'] = subtoken_begin

            token_end = end_to_index[mention['end']]
            subtoken_end = word_idx_to_last_subtoken_idx[token_end]
            mention['subtoken_end'] = subtoken_end

            if 'candidates' in mention:
                # since it is end-to-end, no need to keep candidates here ; they will be kept as span candidates for
                # all the document in a separate dict (see below)
                del mention['candidates']
                del mention['scores']

        # assign 'NILL' to the concepts with link in Null
        for concept in json_doc['concepts']:
            if 'link' in concept and concept['link'] is None:
                concept['link'] = 'NILL'

        # now produces all the candidates for the content and links to span begin and span end on subtoken level
        n_tokens = len(begin)
        span_begin, span_end = create_all_spans(1, n_tokens, args.max_span_length)
        lengths = torch.tensor([n_tokens], dtype=torch.long)
        span_mask = (span_end < lengths.unsqueeze(-1).unsqueeze(-1)).float()

        nr_possible_spans = (span_mask.size(-1) * span_mask.size(-2))

        span_masked_scores = span_mask.view(span_mask.size(0), -1)

        top_indices_sorted = torch.arange(0, nr_possible_spans, dtype=torch.int32).unsqueeze(0)

        all_possible_spans = indices_to_spans(top_indices_sorted,
                                              torch.tensor([nr_possible_spans],
                                                           dtype=torch.int), args.max_span_length)[0]

        linker_cands_all_spans, linker_cands_all_spans_scores = \
            self.get_linker_candidates_all_spans(input_content=json_doc['content'],
                                                 # json_doc['tags'],
                                                 all_spans=all_possible_spans, begin=begin, end=end,
                                                 span_mask=span_masked_scores[0])

        # passes the spans from token (word) level to subtoken (bert) level
        l_span_mask = span_masked_scores[0].tolist()

        spans_data = []

        # Split documents
        map_subtoks_to_segmented_subtoks = self.split_into_segments(bert_doc, bert_doc.sentence_end, bert_doc.token_end)

        for curr_mention in json_doc['mentions']:
            diff_orig = curr_mention['subtoken_end'] - curr_mention['subtoken_begin']
            sn_b = map_subtoks_to_segmented_subtoks[curr_mention['subtoken_begin']]
            sn_e = map_subtoks_to_segmented_subtoks[curr_mention['subtoken_end']]
            if sn_e - sn_b == diff_orig:
                curr_mention['subtoken_begin'] = sn_b
                curr_mention['subtoken_end'] = sn_e
            else:
                # TODO: SHOULD I IGNORE THIS???
                logger.warning('THIS CAN BE SERIOUS MENTION NOT IN THE SAME SEGMENT '
                               '(SPLITTED ACROSS DIFFERENT SEGMENTS): ' +
                               str(bert_doc.subtokens[curr_mention['subtoken_begin']:curr_mention['subtoken_end'] + 1]))
                curr_mention['subtoken_begin'] = sn_b
                curr_mention['subtoken_end'] = sn_e

        # multi-line version (easier to debug)
        for (t1, t2), mask, cands, scores_cands in \
                zip(all_possible_spans, l_span_mask, linker_cands_all_spans, linker_cands_all_spans_scores):
            if mask > 0.9:
                if t1 not in word_idx_to_first_subtoken_idx or t2 not in word_idx_to_last_subtoken_idx:
                    if t1 not in word_idx_to_first_subtoken_idx:
                        logger.warning('problem passing to subtoken ids with the following t1 token: "' +
                                       json_doc['tokenization']['tokens'][t1] + '", cands: ' + str(
                            cands) + ', OMITTING')
                    if t2 not in word_idx_to_first_subtoken_idx:
                        logger.warning('problem passing to subtoken ids with the following t2 token: "' +
                                       json_doc['tokenization']['tokens'][t2] + '", cands: ' + str(
                            cands) + ', OMITTING')
                else:
                    subt_t1 = word_idx_to_first_subtoken_idx[t1]
                    subt_t2 = word_idx_to_last_subtoken_idx[t2]
                    m_subt_t2 = map_subtoks_to_segmented_subtoks[subt_t2]
                    m_subt_t1 = map_subtoks_to_segmented_subtoks[subt_t1]
                    if subt_t2 - subt_t1 == m_subt_t2 - m_subt_t1:
                        spans_data.append(((m_subt_t1, m_subt_t2), cands, scores_cands))
                    else:
                        # TODO: SHOULD WE IGNORE THIS???
                        spans_data.append(((m_subt_t1, m_subt_t2), cands, scores_cands))
                        # logger.warning('FOLLOWING SPAN across segments: ' + str(bert_doc.subtokens[subt_t1:subt_t2 + 1]))

        all_possible_spans = [(t1, t2) for (t1, t2), _, _ in spans_data]

        # has to convert to torch.tensor because it is needed in this format inside get_linker_targets_all_spans
        linker_cands_all_spans = [torch.tensor(l, dtype=torch.int) for (_, _), l, _ in spans_data]
        linker_cands_all_spans_scores = [ls for (_, _), _, ls in spans_data]
        linker_targets_all_spans = self.get_linker_targets_all_spans(json_doc, all_possible_spans,
                                                                     linker_cands_all_spans)

        bert_doc.all_spans = all_possible_spans

        bert_doc.all_spans_candidates = [t.tolist() for t in linker_cands_all_spans]
        bert_doc.all_spans_candidates_scores = linker_cands_all_spans_scores
        bert_doc.all_spans_candidates_targets = linker_targets_all_spans

        bert_doc.mentions = json_doc['mentions']

        document = bert_doc.convert_to_json()
        return document

    def bert_preprocess(self):
        self.documents = list()
        self.tokenizer = get_tokenizer(args.tokenizer_name)
        input_dir = args.input_dir

        for (dirpath, dirnames, filenames) in os.walk(input_dir):
            for curr_file in filenames:
                if 'DW_' in curr_file:
                    logger.info('processing ' + curr_file)
                    parsed_file = json.load(open(os.path.join(dirpath, curr_file)))
                    curr_doc = self.get_document(parsed_file)

                    self.documents.append(curr_doc)

    def save_to_disk(self):
        logger.info('saving to disk in DWIE format please wait...')
        output_dir = args.output_dir

        for curr_doc in self.documents:
            file_name = '{}.json'.format(curr_doc['id'])
            json.dump(curr_doc, open(os.path.join(output_dir, 'dwie_bert_s{}'.format(args.max_seg_len),
                                                  file_name), 'w', encoding='utf8'), ensure_ascii=False)

        logger.info('finished saving to disk in DWIE format')

    def save_to_disk_hoi_format(self):
        """Saves to disk, but in the format that can be used as input to https://github.com/lxucs/coref-hoi"""
        logger.info('saving to disk in hoi format please wait...')
        output_dir = args.output_dir_hoi
        path_dev = os.path.join(output_dir, 'dwie.dev.english.{}.jsonlines'.format(args.max_seg_len))
        path_train = os.path.join(output_dir, 'dwie.train.english.{}.jsonlines'.format(args.max_seg_len))
        out_file_dev = open(path_dev, 'w')
        out_file_train = open(path_train, 'w')

        for curr_doc in self.documents:
            cluster_to_mention = dict()
            struct_output = dict()  # output to be saved to disk
            # passes clusters to hoi format
            struct_output['doc_key'] = curr_doc['id']
            for curr_mention in curr_doc['mentions']:
                cluster_id = curr_mention['concept']
                if cluster_id not in cluster_to_mention:
                    cluster_to_mention[cluster_id] = []
                cluster_to_mention[cluster_id].append([curr_mention['subtoken_begin'], curr_mention['subtoken_end']])

            struct_output['speakers'] = []
            for curr_sentence in curr_doc['bert_segments']:
                curr_speakers = ['-' for _ in curr_sentence]
                curr_speakers[0] = self.speaker_separator
                curr_speakers[-1] = self.speaker_separator
                struct_output['speakers'].append(curr_speakers)

            sorted_clusters = sorted(cluster_to_mention.items(), key=lambda x: x[0])
            sorted_clusters = [x for _, x in sorted_clusters]
            struct_output['clusters'] = sorted_clusters
            struct_output['constituents'] = []
            struct_output['ner'] = []
            struct_output['pronouns'] = []
            struct_output['sentences'] = curr_doc['bert_segments']
            struct_output['sentence_map'] = curr_doc['sentence_map']
            struct_output['subtoken_map'] = curr_doc['subtoken_map']
            struct_output['tokens'] = curr_doc['word_tokenization']
            if 'train' in curr_doc['tags']:
                out_file_train.write(json.dumps(struct_output) + '\n')
            else:
                out_file_dev.write(json.dumps(struct_output) + '\n')

        logger.info('Finished saving to disk in hoi format')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-cased',
                        help='Name or path of the tokenizer/vocabulary')
    parser.add_argument('--input_dir', type=str,
                        default='data/dwie/plain_format/data/annos_with_content/',
                        # default='data/local_tests/dwie-original-1',
                        help='Input directory that contains DWIE files')
    parser.add_argument('--output_dir', type=str, default='data/dwie/spanbert_format/',
                        help='Output directory')
    parser.add_argument('--output_dir_hoi', type=str, default='data/dwie/spanbert_format_hoi/',
                        help='Output directory')
    parser.add_argument('--alias_table_path', type=str,
                        default='data/dwie/dwie-alias-table/dwie-alias-table.json',
                        help='Path to alias file that contains spans to candidate list. ')
    parser.add_argument('--max_nr_candidates', type=int, default=16,
                        help='Maximum nr of candidates to take into account. ')
    parser.add_argument('--max_span_length', type=int, default=5,
                        help='Maximum length (in words) for candidates. ')
    parser.add_argument('--max_seg_len', type=int, default=384,
                        help='Segment length: 128, 256, 384, 512')

    args = parser.parse_args()
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'dwie_bert_s{}'.format(args.max_seg_len)), exist_ok=True)
    links_dictionary: Dictionary = Dictionary()

    links_dictionary_path = os.path.join(args.output_dir, 'links_dictionary.json')
    candidate_links_path = os.path.join(args.output_dir, 'candidate_links.pickle')

    if os.path.isfile(links_dictionary_path) and os.path.isfile(candidate_links_path):
        candidate_links = pickle.load(open(candidate_links_path, 'rb'))
        links_dictionary.load_json(links_dictionary_path)
    else:
        candidate_links, links_dictionary = load_linking_candidates(args.alias_table_path,
                                                                    args.max_nr_candidates,
                                                                    links_dictionary=links_dictionary)
        pickle.dump(candidate_links, open(candidate_links_path, 'wb'))
        links_dictionary.write(links_dictionary_path)

    bert_processor = BertProcessor(args, candidate_links, links_dictionary)

    bert_processor.bert_preprocess()

    bert_processor.save_to_disk()

    # bert_processor.save_to_disk_hoi_format()
