# the (original) goal of this class is to use the output from third_party/kolitsas_e2e/prepro_util.py to produce
# the file that can be used to train the coreflinker (SpanBert version) module (traintool.py).
# Conceptually, it does more or less the same than main_bert_processor_dwie.py, BUT, instead of producing the
# candidates (using the candidate list such as cpn-alias-table.json, it already takes as input the aida
# parsed file with candidates using the Kolitsas paper tool (third_party/kolitsas_e2e/prepro_util.py).
# Which was adapted specifically to produce the json output similar to dwie ones. It also takes as input the
# newly annotated AIDA files (where we also annotate NIL clusters as well as enrich incomplete links). At the
# moment of writing this, the newly annotate aida files are located in the following directory:
# projectcpn/dwie_linker/data/aida/aida_reannotated/aida-20210402/current.

import argparse
import json
import logging
import os
from typing import List, Dict

import torch
from transformers import BertTokenizer

from data_processing.bert_preprocessing import normalize_word, flatten
from data_processing.dictionary import Dictionary
from main_bert_processor_dwie import JsonBertDocument, get_tokenizer
from models.misc.spanner import create_all_spans
from models.utils.misc import indices_to_spans

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def from_kolitsas_to_aida_link_path(filename: str, docid: str):
    """
    From a particular filename and docid produced the aida path.
    Some of the code parts were adapted from external/datasets/find_inconsistencies_reannotated_aida.py.

    :param filename: ex: aida_train.jsonl.
    :param docid: ex: 4_China
    :return: ex: data/aida/aida_reannotated/aida-20210402/current/123.json.

    """

    # different filename possibilities:
    #   1- aida_dev.jsonl
    #       - ex docid: 947testa_CRICKET
    #   2- aida_test.jsonl
    #       - ex docid: 1163testb_SOCCER
    #   3- aida_train.jsonl
    #       - ex docid: 1_EU4
    # to_ret_id = -1
    if filename == 'aida_dev.jsonl':
        ret_id = docid[:docid.index('testa')]
    elif filename == 'aida_test.jsonl':
        ret_id = docid[:docid.index('testb')]
    elif filename == 'aida_train.jsonl':
        ret_id = docid[:docid.index('_')]
    else:
        raise RuntimeError('Unknown filename in from_kolitsas_to_aida_path: ', filename)
    to_ret_id = '{}.json'.format(ret_id)
    return to_ret_id


def from_kolitsas_to_aida_plain_dwie_path(filename: str, docid: str):
    if filename == 'aida_dev.jsonl':
        dc_id = docid[:docid.index('testa')]
        ret_id = 'testa/{}.json'.format(dc_id)
    elif filename == 'aida_test.jsonl':
        dc_id = docid[:docid.index('testb')]
        ret_id = 'testb/{}.json'.format(dc_id)
    elif filename == 'aida_train.jsonl':
        dc_id = docid[:docid.index('_')]
        ret_id = 'train/{}.json'.format(dc_id)
    else:
        raise RuntimeError('Unknown filename in from_kolitsas_to_aida_path: ', filename)

    return ret_id


class BertAidaProcessor(object):
    def __init__(self, args, links_dictionary: Dictionary):
        self.args = args
        # self.linking_candidates: Dict = linking_candidates
        self.documents: List[Dict] = []
        self.tokenizer: BertTokenizer = None
        self.links_dictionary: Dictionary = links_dictionary

        # maps wrongly detected encoding cases of candidates
        self.hard_encoding_cases = {
            'FC_Ceahlăul_Piatra_Neamţ': 'FC_Ceahlăul_Piatra_Neamț',
            'François_Pienaar': 'Francois_Pienaar',
            'Timişoara': 'Timișoara',
            'Ovidiu_Stângă': 'Ovidiu_Stîngă',
            'Anton_Doboş': 'Anton_Doboș',
            'Victor_Babeş': 'Victor_Babeș',
            'Chişinău': 'Chișinău',
            'Dănuţ_Lupu': 'Dănuț_Lupu',
            'FC_Oţelul_Galaţi': 'FC_Oțelul_Galați',
            'FC_Sportul_Studenţesc_Bucureşti': 'FC_Sportul_Studențesc_București',
            'CS_Jiul_Petroşani': 'CS_Jiul_Petroșani',
            'FC_Dinamo_Bucureşti': 'FC_Dinamo_București',
            'FC_Rapid_Bucureşti': 'FC_Rapid_București',
            'FC_Steaua_Bucureşti': 'FC_Steaua_București',
            'Constanţa': 'Constanța',
            'Bistriţa': 'Bistrița'
        }

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

    def get_linker_candidates_all_spans(self, all_spans, begin, end, json_cand_doc, tokens):

        span_to_candidates = dict()
        for span_begin, span_end, candidates, candidate_scores, span_text in zip(json_cand_doc['span_begin'],
                                                                                 json_cand_doc['span_end'],
                                                                                 json_cand_doc['candidates'],
                                                                                 json_cand_doc['scores'],
                                                                                 json_cand_doc['spans']):
            span_to_candidates[(span_begin, span_end)] = dict()
            if len(candidates) != len(set(candidates)):
                # removes the duplicates, starting from the back (the ones with lower candidate score
                new_candidates = list()
                new_candidate_scores = list()
                candidates_added = set()
                for curr_candidate, curr_cand_score in zip(candidates, candidate_scores):
                    if curr_candidate not in candidates_added:
                        new_candidates.append(curr_candidate)
                        new_candidate_scores.append(curr_cand_score)

                        candidates_added.add(curr_candidate)
                candidates = new_candidates
                candidate_scores = new_candidate_scores

            span_to_candidates[(span_begin, span_end)]['candidates'] = candidates
            span_to_candidates[(span_begin, span_end)]['scores'] = candidate_scores
            span_to_candidates[(span_begin, span_end)]['span_text'] = span_text

        candidates = []
        candidates_scores = []
        for idx_span, curr_span in enumerate(all_spans):

            span_candidates = []
            span_scores = []
            if (curr_span[0], curr_span[1] + 1) in span_to_candidates:
                curr_cand_data = span_to_candidates[(curr_span[0], curr_span[1] + 1)]
                span_candidates = []

                for cnd in curr_cand_data['candidates']:
                    decoded_candidate = cnd.encode('ascii', 'backslashreplace').decode('unicode-escape')
                    if decoded_candidate in self.hard_encoding_cases:
                        decoded_candidate = self.hard_encoding_cases[decoded_candidate]
                    dict_added = self.links_dictionary.add(decoded_candidate)
                    span_candidates.append(dict_added)
                span_scores = curr_cand_data['scores']
                span_text = ' '.join(tokens[curr_span[0]: curr_span[1] + 1])
                cand_text = curr_cand_data['span_text']
                # checks that our the span text is equal to the span text that produced the original candidates
                # (using prepro_util.py)
                assert span_text == cand_text

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

            # the TARGET links sometimes come with escaped characters,
            correct_link = correct_link.encode('ascii', 'backslashreplace').decode('unicode-escape')
            if correct_link in self.hard_encoding_cases:
                correct_link = self.hard_encoding_cases[correct_link]

            correct_link_id = self.links_dictionary.add(correct_link)
            target_index = (all_spans_candidates[span_idx] == correct_link_id).nonzero()
            if target_index.size()[0] == 0:
                index = -1
            else:
                index = target_index[0].item()

            targets.append(index)

        return targets

    def get_document(self, json_doc, json_cand_doc) -> Dict:
        """ Process raw input to finalized documents """
        bert_doc: JsonBertDocument = JsonBertDocument(json_doc, args=self.args)
        word_idx = -1

        begin = json_doc['tokenization']['begin']
        end = json_doc['tokenization']['end']
        sentences = json_doc['tokenization']['sentences']

        bert_doc.begin_token = begin
        bert_doc.end_token = end

        begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
        end_to_index = {pos: idx for idx, pos in enumerate(end)}

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
            for idx_subtoken, subtoken in enumerate(subtokens):
                bert_doc.subtokens.append(subtoken)
                if idx_subtoken == 0:
                    word_idx_to_first_subtoken_idx[word_idx] = subtoken_idx
                word_idx_to_last_subtoken_idx[word_idx] = subtoken_idx
                if idx_token >= len(json_doc['tokenization']['tokens']) - 1 and idx_subtoken >= len(subtokens) - 1:
                    # the last subtoken in the document always is the end of sentence by definition
                    bert_doc.sentence_end.append(True)
                elif idx_token < len(json_doc['tokenization']['tokens']) - 1 and \
                        sentences[idx_token + 1] != sentences[idx_token] and \
                        idx_subtoken >= len(subtokens) - 1:
                    bert_doc.sentence_end.append(True)
                else:
                    # in all other cases, it is not the end of sentence
                    bert_doc.sentence_end.append(False)
                bert_doc.subtoken_map.append(word_idx)
                subtoken_idx += 1

        assert len(bert_doc.subtokens) == len(bert_doc.sentence_end)
        # now maps all the mentions to the bert subtoken positions
        for mention in json_doc['mentions']:
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
                if 'scores' in mention:
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

        # json_doc['content'] should be used for debugging purposes only!!!
        linker_cands_all_spans, linker_cands_all_spans_scores = \
            self.get_linker_candidates_all_spans(all_spans=all_possible_spans, begin=begin, end=end,
                                                 json_cand_doc=json_cand_doc, tokens=json_doc['tokenization']['tokens'])

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
                        spans_data.append(((m_subt_t1, m_subt_t2), cands, scores_cands))

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

        aida_plain_dwie_path = args.aida_plain_dwie_path
        for (dirpath, dirnames, filenames) in os.walk(aida_candidate_path):
            for curr_filename in filenames:
                if curr_filename.endswith('.jsonl'):
                    with open(os.path.join(dirpath, curr_filename)) as infile:
                        for curr_line in infile:
                            parsed_cand = json.loads(curr_line)
                            logger.info('Processing: ' + parsed_cand['doc_id'])
                            aida_link_file = from_kolitsas_to_aida_link_path(curr_filename, parsed_cand['doc_id'])
                            aida_link_path = os.path.join(args.aida_link_path, aida_link_file)
                            parsed_aida_reannotated_johannes_tokenization = json.load(open(aida_link_path))

                            aida_plain_dwie_file = from_kolitsas_to_aida_plain_dwie_path \
                                (curr_filename, parsed_cand['doc_id'])

                            aida_plain_path = os.path.join(aida_plain_dwie_path, aida_plain_dwie_file)

                            parsed_aida_original_tokenization = json.load(open(aida_plain_path))

                            # needed to extract the candidates based on the token ids
                            char_pos_begin_to_tok_idx = {char_pos: tok_idx for tok_idx, char_pos in
                                                         enumerate(parsed_aida_original_tokenization
                                                                   ['tokenization']['begin'])}
                            char_pos_end_to_tok_idx = {char_pos: tok_idx + 1 for tok_idx, char_pos in
                                                       enumerate(parsed_aida_original_tokenization
                                                                 ['tokenization']['end'])}

                            aida_plain_mentions = parsed_aida_original_tokenization['mentions']
                            aida_link_mentions = parsed_aida_reannotated_johannes_tokenization['mentions']

                            assert len(aida_plain_mentions) == len(aida_link_mentions)

                            tok_idxs_to_candidates = dict()
                            for span_begin, span_end, candidates in zip(parsed_cand['span_begin'],
                                                                        parsed_cand['span_end'],
                                                                        parsed_cand['candidates']):
                                decoded_candidates = list()
                                for cnd in candidates:
                                    decoded_candidate = cnd.encode('ascii', 'backslashreplace').decode('unicode-escape')
                                    if decoded_candidate in self.hard_encoding_cases:
                                        decoded_candidate = self.hard_encoding_cases[decoded_candidate]
                                    decoded_candidates.append(decoded_candidate)
                                tok_idxs_to_candidates[(span_begin, span_end)] = decoded_candidates

                            for mention_plain, mention_link in zip(aida_plain_mentions, aida_link_mentions):
                                assert mention_plain['text'] == mention_link['text']
                                assert mention_plain['begin'] == mention_link['begin']
                                assert mention_plain['end'] == mention_link['end']
                                tok_idx_begin = char_pos_begin_to_tok_idx[mention_plain['begin']]
                                tok_idx_end = char_pos_end_to_tok_idx[mention_plain['end']]
                                if (tok_idx_begin, tok_idx_end) in tok_idxs_to_candidates:
                                    mention_plain['candidates'] = tok_idxs_to_candidates[(tok_idx_begin, tok_idx_end)]
                                mention_plain['concept'] = mention_link['concept']

                            parsed_aida_original_tokenization['concepts'] = \
                                parsed_aida_reannotated_johannes_tokenization['concepts']

                            # make sure the link is correctly encoded for each of the concepts
                            for curr_concept in parsed_aida_original_tokenization['concepts']:
                                if 'link' in curr_concept and curr_concept['link'] is not None:
                                    curr_concept['link'] = curr_concept['link'] \
                                        .encode('ascii', 'backslashreplace').decode('unicode-escape')
                                    if curr_concept['link'] in self.hard_encoding_cases:
                                        curr_concept['link'] = self.hard_encoding_cases[curr_concept['link']]

                            # saves to json with updated linking annotations
                            output_plain_path = os.path.join(output_dir_aida_tok, aida_plain_dwie_file)

                            os.makedirs(os.path.dirname(output_plain_path), exist_ok=True)

                            json.dump(parsed_aida_original_tokenization, open(output_plain_path, 'w'), indent=4)

                            curr_doc = self.get_document(parsed_aida_original_tokenization, parsed_cand)

                            curr_doc['tags'] = ['all']
                            if 'train' in aida_plain_dwie_file:
                                curr_doc['tags'].append('train')
                            elif 'testb' in aida_plain_dwie_file:
                                curr_doc['tags'].append('testb')
                            elif 'testa' in aida_plain_dwie_file:
                                curr_doc['tags'].append('testa')
                            else:
                                raise RuntimeError('can not find the dataset type in aida_plain_dwie_file: ' +
                                                   aida_plain_dwie_file)

                            self.documents.append(curr_doc)

    def save_to_disk(self):
        for curr_doc in self.documents:
            file_name = '{}.json'.format(curr_doc['id'])
            if 'train' in curr_doc['tags']:
                set_type = 'train'
            elif 'testa' in curr_doc['tags']:
                set_type = 'testa'
            elif 'testb' in curr_doc['tags']:
                set_type = 'testb'
            else:
                raise RuntimeError('can not identify the set_type in tags: ' + str(curr_doc['tags']))
            json.dump(curr_doc, open(os.path.join(output_dir_spanbert, set_type, file_name), 'w'))
        # also saves the link dictionary
        self.links_dictionary.write(os.path.join(output_dir_spanbert, 'links_dictionary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aida_link_path', type=str,
                        default='data/aida/aida_reannotated/aida-20210402/current',
                        help='The newly annotated aida links using the following annotation tool: '
                             'http://10.10.3.23:8083/browser/overlord/051_aida/train/list?view=output.highlighted'
                             'and then exported by Johannes into dwie format.')

    parser.add_argument('--aida_candidate_path', type=str,
                        default='third_party/kolitsas_e2e/data/json_records/corefmerge/allspans',
                        help='The file with link candidates for aida produced by the adapted script originally '
                             'downloaded from the github of the work of Kolitsas et al. The script is in '
                             'third_party/kolitsas_e2e/prepro_util.py.')

    parser.add_argument('--aida_plain_dwie_path', type=str,
                        default='data/aida/aida/partitioned/aida_dwie_format_no_coref_annos',
                        help='Files generated by s1_corpus_extractor.py with the aida original tokenization. '
                             'We need it to pass from the tokens in aida_candidate_path to the links in '
                             'aida_links_path through (hopefully) the char positions in the original text. ')

    parser.add_argument('--output_dir', type=str,
                        default='data/aida/aida_reannotated/aida-20210402/transformed')

    parser.add_argument('--tokenizer_name', type=str, default='bert-base-cased',
                        help='Name or path of the tokenizer/vocabulary')

    parser.add_argument('--max_span_length', type=int, default=10, help='Maximum width of the span, the default is 10 '
                                                                        'because this is also the value'
                                                                        'used in the prepro_util.py of Kolitsas et al.')

    parser.add_argument('--max_seg_len', type=int, default=384,
                        # parser.add_argument('--max_seg_len', type=int, default=256,
                        # parser.add_argument('--max_seg_len', type=int, default=384,
                        help='Segment length: 128, 256, 384, 512')

    args = parser.parse_args()

    output_dir_aida_tok = os.path.join(args.output_dir, 'aida_tokenization_adaptation')

    output_dir_spanbert = os.path.join(args.output_dir, 'spanbert_s{}'.format(args.max_seg_len))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_dir_aida_tok, exist_ok=True)
    os.makedirs(output_dir_spanbert, exist_ok=True)
    os.makedirs(os.path.join(output_dir_spanbert, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_spanbert, 'testa'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_spanbert, 'testb'), exist_ok=True)

    aida_candidate_path = args.aida_candidate_path

    # dictionary, already preloads NONE and NILL
    aida_link_dictionary: Dictionary = Dictionary()

    aida_link_dictionary.add('NILL')

    bert_processor = BertAidaProcessor(args, links_dictionary=aida_link_dictionary)

    bert_processor.bert_preprocess()

    bert_processor.save_to_disk()
