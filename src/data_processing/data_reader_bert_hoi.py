# (kzaporoj 09/04/2021) - This is a very similar to data_reader_bert.py, BUT the main idea is to produce as much
# output in tensor format as possible; avoid the use of lists that later on get converted to tensors inside of
# the collate_dygie_spanbert function when the model is training. The reason is that having tensors already should
# make it faster.
import json
import logging
import os
import random
from time import sleep

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_processing.dictionary import Dictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class DatasetDWIESpanBertHoi(Dataset):
    def __init__(self, name, config, dictionaries):
        super().__init__()
        self.name = name
        self.tag = config['dataset']['tag']
        self.bert_max_segment_len = config['dataloader']['bert_max_segment_len']
        self.instances = []

        self.include_nill_in_candidates = config['dataloader']['include_nill_in_candidates']
        self.include_none_in_candidates = config['dataloader']['include_none_in_candidates']

        self.bert_dictionary = dictionaries['bert_subtokens']
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

        path = config['dataset']['filename']
        logging.info('Loading {} tag={}'.format(path, self.tag))

        if os.path.isdir(path):
            for filename in tqdm(os.listdir(path)):
                f = os.path.join(path, filename)
                self.load_file(f)

        else:
            self.load_file(path)

    def get_linker_gold_from_mentions(self, data):
        gold = []
        for mention in data['mentions']:
            assert mention['subtoken_begin'] is not None
            assert mention['subtoken_end'] is not None

            concept = data['concepts'][mention['concept']]

            if 'link' in concept:
                gold.append((mention['subtoken_begin'], mention['subtoken_end'], concept['link']))

        return gold

    def convert(self, data):
        identifier = data['id']
        concepts = data['concepts']

        subtoken_map = data['subtoken_map']
        token_length = max(subtoken_map) + 1  # + 1 because zero-based
        bert_segments = data['bert_segments']

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

        to_ret = {
            'id': identifier,
            'metadata_tags': data['tags'],
            'bert_segments': bert_segms,
            'bert_segs_mask': bert_segs_mask,
            'num_concepts': len(gold_clusters),
            'gold_clusters': gold_clusters,
            'clusters': torch.IntTensor([mention['concept'] for mention in mentions]),
            'gold_subtokens_start': torch.IntTensor([s[0] for s in spans]),
            'gold_subtokens_end': torch.IntTensor([s[1] for s in spans]),
            'all_spans_tensor': torch.tensor(f_all_spans),
            'linker_candidates': f_linker_candidates,
            'linker_targets': f_linker_targets,
            'linker_scores': f_linker_scores,
            'begin_token': data['begin_token'],
            'end_token': data['end_token'],
            'linker_gold': linker_gold,
            'subtoken_map': subtoken_map,
            'token_length': token_length,
            'sentence_map': torch.IntTensor(data['sentence_map'])
        }

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
                    logging.warning('following exept: %s' % exept.strerror)
                    logging.warning('except to load, trying again: %s for retry: %s' % (filename, retry))
                    sleep(random.randint(5, 10))
                    self.load_file(filename, retry=retry + 1)
                else:
                    logging.error('NO MORE TERIES LEFT, FAILING')
                    raise exept
        elif filename.endswith('.jsonl'):
            self.load_jsonl(filename)
        else:
            raise BaseException('unknown file type: %s' % filename)

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            if 'tags' in data:
                if self.tag in data['tags']:
                    self.instances.append(self.convert(data))
            else:
                logging.warning('WARNING (kzaporoj) - NO tags IN %s' % filename)

    def load_jsonl(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                data = json.loads(line.rstrip())
                if self.tag in data['tags']:
                    self.instances.append(self.convert(data))

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)
