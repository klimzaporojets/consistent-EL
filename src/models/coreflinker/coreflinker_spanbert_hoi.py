"""
The original idea of this module is to merge some ideas from hoi project (https://github.com/lxucs/coref-hoi)
It requires some changes that can affect the way we represent the candidates or mask, dropout scheme, etc. for example,
this is why I decided to put it in a separate module.
It will probably entail the creation of separate versions "hoi" of other submodules such as pruner, span extractor, etc.
For now the goal is to make it work with batch size of 1 which is consistent with https://github.com/lxucs/coref-hoi
and also with the memory requirements of span-based model.

"""
from typing import Dict

import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

import settings
from cpn.builder import convert_to_json
# from models.dygie import collate_dygie
from models.coreflinker.attention import ModulePlainAttention
from models.coreflinker.attprop import ModuleAttentionPropHoi
from models.coreflinker.corefbasic import ModuleCorefBasic
from models.coreflinker.coreflinker_mtt_prop_hoi import ModuleCorefLinkerMTTPropE2EHoi
from models.coreflinker.coreflinker_prop import ModuleCorefLinkerDisabled
from models.coreflinker.coreflinker_prop_hoi import ModuleCorefLinkerPropE2EHoi
from models.coreflinker.corefprop import ModuleCorefProp
from models.coreflinker.corefprop2 import ModuleCorefProp2
from models.coreflinker.corefprop_hoi import ModuleCorefPropHoi
from models.coreflinker.pruner import MentionPrunerSpanBertHoi
from models.coreflinker.relbasic import ModuleRelBasic
from models.coreflinker.relprop import ModuleRelProp, ModuleRelPropX
from models.coreflinker.relprop1 import ModuleRelProp1
from models.coreflinker.relprop2 import ModuleRelProp2
from models.coreflinker.relsimple import ModuleRelSimple
from modules.misc.misc import batched_index_select
from modules.ner.spanner import SpanEndpointSpanBert
from modules.spirit import TextEmbedder
from modules.tasks.coref import LossCoref, LossBidirectionalCoref
from modules.tasks.coreflinker import CorefLinkerLossDisabled
from modules.tasks.coreflinker_edmonds_softmax import LossCorefLinkerESM
from modules.tasks.coreflinker_hoi import CorefLinkerLossHoi
from modules.tasks.coreflinker_mtt_hoi import LossCorefLinkerMTTHoi
from modules.tasks.linker import LinkerNone, collate_candidates_in_pytorch, \
    collate_targets
from modules.tasks.linker_hoi import LossLinkerE2EHoi
from modules.utils.misc import inspect, spans_to_indices, SpanPairs


def collate_dygie_spanbert(model, batch, device, collate_api=False):
    """

    :param model:
    :param batch:
    :param device:
    :param collate_api: if in True, means that the input comes from a client, possibly as a free text
    (i.e., no gold mentions, relations, concepts, spans, etc.). If in False (default), the input comes for training
    or evaluating using internal function located in traintool.train for instance.
    :return:
    """
    assert len(batch) == 1  # module for only batch size 1
    sequence_lengths = [x['bert_segs_mask'].sum().item() for x in batch]
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int, device=settings.device)

    tokens_lengths = torch.tensor([x['token_length'] for x in batch], dtype=torch.int, device=settings.device)
    inputs = {
        'bert_segments': [x['bert_segments'].to(device=settings.device) for x in batch],
        'bert_segments_mask': [x['bert_segs_mask'].to(device=settings.device) for x in batch],
        'sequence_lengths': sequence_lengths,
        'token_lengths': tokens_lengths
        # 'characters': characters.to(device) if characters is not None else None,
        # 'sequence_lengths': sequence_lengths.to(device),
        # 'token_indices': indices.to(device),
        # 'text': [b['xxx']['text'] for b in batch]
    }

    # gold_spans = [[(m[0], m[1]) for m in x['spans']] for x in batch]

    # if not collate_api:
    # if 'gold_clusters' in batch[0]:
    #     gold_clusters = [x['gold_clusters'] for x in batch]
    # else:
    #     # TODO: move to cpn utility .py (or remove)
    #     gold_clusters = []
    #     for spans, m2c in zip(gold_spans, [x['mention2concept'] for x in batch]):
    #         clusters = [list() for _ in range(m2c[0])]
    #         for mention, concept in zip(m2c[3], m2c[2]):
    #             clusters[concept].append(spans[mention])
    #         gold_clusters.append(clusters)
    # end if not collate_api:

    metadata = {
        # 'tokens': [x['xxx']['text'] for x in batch],
        'content': [x['content'] for x in batch],
        # 'begin': [[] for _ in batch],
        # 'end': [[] for _ in batch]
        'begin_token': [x['begin_token'] for x in batch],
        'end_token': [x['end_token'] for x in batch],
        'subtoken_map': [x['subtoken_map'] for x in batch],
        'sentence_map': [x['sentence_map'] for x in batch],
    }

    # if not collate_api:
    metadata['identifiers'] = [x['id'] for x in batch]
    metadata['tags'] = [x['metadata_tags'] for x in batch]
    # end if not collate_api:

    # relations = None
    # if not collate_api:
    # metadata['gold_tags_indices'] = [x['gold_tags_indices'] for x in batch]
    # metadata['all_spans']

    # metadata['gold_spans'] = gold_spans

    # metadata['gold_spans_lengths'] = (torch.LongTensor([len(curr_spans) for curr_spans in gold_spans])).to(
    #     device=settings.device)
    metadata['gold_m2i'] = [x['clusters'] for x in batch]

    # relations = {
    #     # 'gold_spans': gold_spans,
    #     'gold_m2i': [x['clusters'] for x in batch],
    #     'gold_clusters2': gold_clusters
    # }
    #
    # if 'relations' in batch[0]:
    #     # old: remove the dimension
    #     relations['gold_relations'] = [x['relations'][1] for x in batch]
    #     relations['num_concepts'] = [x['relations'][0][0] for x in batch]
    # else:
    #     relations['gold_relations'] = [x['relations2'] for x in batch]
    #     relations['num_concepts'] = [x['num_concepts'] for x in batch]
    # gold_spans = None
    # spans_tensors = collate_spans(gold_spans)
    # spans_tensors = None
    gold_subtokens_start = batch[0]['gold_subtokens_start']
    gold_subtokens_end = batch[0]['gold_subtokens_end']
    spans_tensors = torch.cat([gold_subtokens_start.unsqueeze(0), gold_subtokens_end.unsqueeze(0)]).T.unsqueeze(0)
    # spans_tensors = spans_tensors.to(device-settings.device)
    metadata['gold_spans_tensors'] = spans_tensors.to(device=settings.device)

    linker = {}
    # linker['all_spans'] = [[(m[0], m[1]) for m in x['all_spans']] for x in batch]
    linker['all_spans_tensor'] = batch[0]['all_spans_tensor'].unsqueeze(0).to(device=settings.device)
    if (model.linker_task.enabled or model.coref_linker_task.enabled) and 'linker_candidates' in batch[0]:
        # or model.coref_linker_mtt_task.enabled)
        candidates, candidate_lengths = collate_candidates_in_pytorch([x['linker_candidates'] for x in batch],
                                                                      unknown_id=model.entity_dictionary.lookup(
                                                                          '###UNKNOWN###'))
        linker['candidates'] = candidates
        linker['candidate_lengths'] = candidate_lengths
        linker['targets'] = collate_targets([x['linker_targets'] for x in batch], candidates.size(2))

        # the spans to which the candidates are assigned in linker

        # linker['total_cand_lengths_in_gold_mentions'] = \
        #     collate_tot_cand_lengths([torch.tensor(x['total_cand_lengths_in_gold_mentions'], dtype=torch.int32)
        #                               for x in batch])

        linker['gold'] = [x['linker_gold'] for x in batch]

        # end if not collate_api:

    metadata['linker'] = linker
    metadata['api_call'] = collate_api

    return {
        'inputs': inputs,
        # 'relations': relations,
        'metadata': metadata
    }


def create_spanprop_hoi(model, config):
    if 'spanprop' in config:
        sp_type = config['spanprop']['type']

        if sp_type == 'attprop':
            return ModuleAttentionPropHoi(model.span_extractor.dim_output,
                                          # model.span_pruner.scorer,
                                          model.span_pair_generator, config['spanprop'])
        else:
            raise BaseException("no such spanprop:", sp_type)
    else:
        return None


def create_corefprop(model, config):
    cp_type = config['corefprop']['type']

    if cp_type == 'none':
        return None
    elif cp_type == 'basic':
        return ModuleCorefBasic(model.span_extractor.dim_output, model.span_pruner.scorer, model.span_pair_generator,
                                config['corefprop'])
    elif cp_type == 'default':
        return ModuleCorefProp(model.span_extractor.dim_output, model.span_pruner.scorer, model.span_pair_generator,
                               config)
    elif cp_type == 'corefprop2':
        return ModuleCorefProp2(model.span_extractor.dim_output, model.span_pruner.scorer, model.span_pair_generator,
                                config['corefprop'])
    elif cp_type == 'attention':
        return ModulePlainAttention(model.span_extractor.dim_output, 1, model.span_pair_generator, config['relprop'],
                                    squeeze=True)
    else:
        raise BaseException("no such corefprop:", cp_type)


def create_coreflinker_prop(model, config, dictionaries):
    if 'coreflinker' not in config or not config['coreflinker']['enabled']:
        return ModuleCorefLinkerDisabled()

    cp_type = config['coreflinker']['coreflinker_prop']['type']

    if cp_type == 'none':
        return None
    elif cp_type == 'default':
        coreflinker_type = config['coreflinker']['type']
        if model.end_to_end_mentions:
            if coreflinker_type == 'coreflinker':
                # return ModuleCorefLinkerPropE2E(model.span_extractor.dim_output,
                return ModuleCorefLinkerPropE2EHoi(model.span_extractor.dim_output,
                                                   model.span_pruner.scorer,
                                                   model.span_pair_generator, config['coreflinker'], dictionaries)
            elif coreflinker_type == 'coreflinker_mtt':
                return ModuleCorefLinkerMTTPropE2EHoi(model.span_extractor.dim_output,
                                                      model.span_pruner.scorer,
                                                      model.span_pair_generator, config['coreflinker'], dictionaries)
            else:
                raise RuntimeError('coreflinker_spanbert_hoi does not recognize the following coreflinker type: ' +
                                   coreflinker_type)
            # elif coreflinker_type == 'coreflinker_esm':
            #     return ModuleCorefLinkerPropE2E(model.span_extractor.dim_output,
            #                                     model.span_pruner.scorer,
            #                                     model.span_pair_generator, config['coreflinker'], dictionaries)
        else:
            raise RuntimeError('coreflinker_spanbert_hoi is not implemented to run in NOT end-to-end mode')
            # if coreflinker_type == 'coreflinker':
            #     return ModuleLinkerCorefProp(model.span_extractor.dim_output,
            #                                  model.span_pruner.scorer,
            #                                  model.span_pair_generator, config['coreflinker'], dictionaries)
            # elif coreflinker_type == 'coreflinker_mtt':
            #     return ModuleCorefLinkerMTTPropE2E(model.span_extractor.dim_output,
            #                                        model.span_pruner.scorer,
            #                                        model.span_pair_generator, config['coreflinker'], dictionaries)
            # elif coreflinker_type == 'coreflinker_esm':
            #     return ModuleLinkerCorefProp(model.span_extractor.dim_output,
            #                                  model.span_pruner.scorer,
            #                                  model.span_pair_generator, config['coreflinker'], dictionaries)
    else:
        raise BaseException("no such linkercoref prop:", cp_type)
    # raise BaseException("no such coreflinker found (in coreflinker_prop):", config['coreflinker'])


def create_coreflinker_loss(model, config):
    if model.coref_linker_scorer.enabled:
        coreflinker_type = config['coreflinker']['type']
        if coreflinker_type == 'coreflinker':
            return CorefLinkerLossHoi('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                      config['coreflinker'], model.end_to_end_mentions)
        elif coreflinker_type == 'coreflinker_mtt':
            return LossCorefLinkerMTTHoi('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                         config['coreflinker'], model.end_to_end_mentions)
        elif coreflinker_type == 'coreflinker_esm':
            return LossCorefLinkerESM('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                      config['coreflinker'], model.end_to_end_mentions)
    else:
        return CorefLinkerLossDisabled()

    # raise BaseException("no such coreflinker found (in create_coreflinker_loss):", config['coreflinker'])


def create_relprop(model, config):
    rp_type = config['relprop']['type']

    if rp_type == 'none':
        return None
    elif rp_type == 'basic':
        return ModuleRelBasic(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                              config['relprop'])
    elif rp_type == 'default':
        return ModuleRelProp(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                             config['relprop'])
    elif rp_type == 'default-x':
        return ModuleRelPropX(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                              config['relprop'])
    elif rp_type == 'simple':
        return ModuleRelSimple(model.span_extractor.dim_output, model.relation_labels, config['relprop'])
    elif rp_type == 'relprop1':
        return ModuleRelProp1(model.span_extractor.dim_output, model.span_pruner, model.span_pair_generator,
                              model.relation_labels, config['relprop'])
    elif rp_type == 'relprop2':
        return ModuleRelProp2(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                              config['relprop'])
    elif rp_type == 'attention':
        return ModulePlainAttention(model.span_extractor.dim_output, len(model.relation_labels),
                                    model.span_pair_generator, config['relprop'])
    else:
        raise BaseException("no such relprop:", rp_type)


class CoreflinkerSpanBertHoi(nn.Module):

    def __init__(self, dictionaries, config):
        super(CoreflinkerSpanBertHoi, self).__init__()
        self.random_embed_dim = config['random_embed_dim']
        self.max_span_length = config['max_span_length']
        # self.hidden_dim = config['hidden_dim']  # 150
        # self.hidden_dp = config['hidden_dropout']  # 0.4
        self.rel_after_coref = config['rel_after_coref']
        self.spans_single_sentence = config['spans_single_sentence']
        self.use_all_subtoken_spans = config['use_all_subtoken_spans']
        # self.spanbert_input = config['spanbert_input']

        # self.load_doc_level_candidates = \
        #     (config['coreflinker']['enabled'] and config['coreflinker']['doc_level_candidates']) or \
        #     (config['linker']['enabled'] and config['linker']['doc_level_candidates'])

        self.debug_memory = False
        self.debug_tensors = False

        # whether take gold mentions or use the pruner
        self.end_to_end_mentions = config['end_to_end_mentions']
        self.embedder = TextEmbedder(dictionaries, config['text_embedder'])

        self.entity_dictionary = None
        if 'entities' in dictionaries:
            self.entity_dictionary = dictionaries['entities']

        if 'lexical_dropout' in config:
            self.emb_dropout = nn.Dropout(config['lexical_dropout'])
        else:
            self.emb_dropout = None
            # self.seq2seq = Seq2Seq(self.embedder.dim_output + self.random_embed_dim, config['seq2seq'])

        # self.span_extractor = create_span_extractor(self.seq2seq.dim_output, self.max_span_length,
        #                                             config['span-extractor'])
        self.span_extractor: SpanEndpointSpanBert = SpanEndpointSpanBert(self.embedder.dim_output,
                                                                         self.max_span_length,
                                                                         config['span-extractor'])
        # create_span_extractor(self.embedder.dim_output,
        #                                                               self.max_span_length,
        #                                                               config['span-extractor'])

        if self.end_to_end_mentions:
            self.span_pruner: MentionPrunerSpanBertHoi = \
                MentionPrunerSpanBertHoi(self.span_extractor.dim_output, config['pruner'],
                                         config['span-extractor']['span_embed'])
        else:
            # self.span_pruner = MentionPrunerGold(self.max_span_length, config['pruner'])
            raise NotImplementedError('not implemented for not end-to-end setting')

        self.span_pair_generator = SpanPairs(self.span_extractor.dim_output, config['span-pairs'])
        self.span_prop = create_spanprop_hoi(self, config)
        # self.span_prop = None

        # self.coref_scorer = create_corefprop(self, config['coref'])
        self.coref_scorer = ModuleCorefPropHoi(self.span_extractor.dim_output, self.span_pruner.scorer,
                                               # self.span_pair_generator,
                                               config, span_pair_generator=self.span_pair_generator)

        # self.relation_labels = dictionaries['relations'].tolist()

        # self.rel_scorer = create_relprop(self, config)

        if not config['coref']['bidirectional']:
            self.coref_task = LossCoref('coref', config['coref'])
        else:
            self.coref_task = LossBidirectionalCoref('coref', config['coref'])

        # kzaporoj - here add linkercoref joint
        self.coref_linker_scorer = create_coreflinker_prop(self, config, dictionaries)
        self.coref_linker_task = create_coreflinker_loss(self, config)

        # self.ner_task = TaskSpan1x('tags', self.span_extractor.dim_output, dictionaries['tags'], config['ner'])
        # self.relation_task = create_task_relations('rels', config['relations'], self.relation_labels)

        if 'linker' in config and config['linker']['enabled']:
            if self.end_to_end_mentions:
                self.linker_task = LossLinkerE2EHoi('links', self.span_extractor.dim_output, dictionaries,
                                                    config['linker'], self.max_span_length)
            else:
                raise RuntimeError('not implemented linker for not end to end')
                # self.linker_task = LossLinker('links', self.span_extractor.dim_output, dictionaries, config['linker'])
        else:
            self.linker_task = LinkerNone()

        if not self.span_pruner.sort_after_pruning and self.pairs.requires_sorted_spans:
            raise BaseException("ERROR: spans MUST be sorted")

    def collate_func(self, datasets, device):
        return lambda x: collate_dygie_spanbert(self, x, device)

    def log_stats(self, dataset_name, tb_logger, step_nr):
        self.span_pruner.log_stats(dataset_name, not self.training, tb_logger, step_nr)
        self.coref_linker_task.log_stats(dataset_name, not self.training, tb_logger, step_nr)
        self.coref_linker_scorer.log_stats(dataset_name, not self.training, tb_logger, step_nr)

    # def end_epoch(self, dataset_name):
    #     self.span_pruner.end_epoch(dataset_name, not self.training)
    #     self.coref_linker_task.end_epoch(dataset_name,self.training)
    #     self.coref_linker_scorer.end_epoch()

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            # if name.startswith('bert'):
            if name.startswith('embedder.spanbert_embedder.spanbert_model'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                # print('name of param: ', name)
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, inputs, metadata, metrics=[], only_loss=False):
        output = {}
        # all_spans_tensor = torch.tensor(metadata['linker']['all_spans'], dtype=torch.int, device=settings.device)
        all_spans_tensor = metadata['linker']['all_spans_tensor']
        # all_spans
        # candidate_starts = all_spans_tensor[:,0]
        # candidate_ends = all_spans_tensor[:,1]
        sequence_lengths = inputs['sequence_lengths']
        token_lengths = inputs['token_lengths']

        if self.debug_memory or self.debug_tensors:
            # print("START", metadata['identifiers'][0], sequence_lengths)
            print("START", metadata['identifiers'][0])
            print("(none)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # MODEL MODULES
        embeddings = self.embedder(inputs)

        if self.emb_dropout is not None:
            embeddings = self.emb_dropout(embeddings)

        if self.random_embed_dim > 0:
            rand_embedding = torch.FloatTensor(embeddings.size(0), embeddings.size(1), self.random_embed_dim).to(
                embeddings.device).normal_(std=4.0)
            rand_embedding = batched_index_select(rand_embedding, inputs['token_indices'])
            embeddings = torch.cat((embeddings, rand_embedding), -1)

        if self.debug_tensors:
            inspect('embeddings', embeddings[0, :, :])

        # we work only with batch size 1 in this module
        assert embeddings.shape[0] == 1

        # embeddings = torch.squeeze(embeddings, 0)
        embeddings = torch.squeeze(embeddings, 0)
        num_subtokens = embeddings.shape[0]

        span_begin = torch.unsqueeze(torch.arange(0, num_subtokens,
                                                  device=settings.device), 1).repeat(1, self.max_span_length)
        span_end = span_begin + torch.arange(0, self.max_span_length, device=settings.device)

        span_begin.unsqueeze_(0)
        span_end.unsqueeze_(0)

        # span_mask = torch.zeros((span_begin.size(0), span_begin.size(1), self.max_span_length),
        #   device=settings.device)
        span_mask = torch.zeros((span_begin.size(0), span_begin.size(1), self.max_span_length),
                                device=settings.device, dtype=torch.bool)

        span_mask[0] = (span_end[0] < num_subtokens)
        if self.spans_single_sentence:
            sentence_map = metadata['sentence_map'][0].to(settings.device)
            assert sentence_map.shape[0] == num_subtokens
            candidate_start_sent_idx = sentence_map[span_begin[0]]
            candidate_end_sent_idx = sentence_map[
                torch.min(span_end[0], torch.tensor(num_subtokens - 1, device=settings.device))]
            span_mask[0] = span_mask[0] & (candidate_start_sent_idx == candidate_end_sent_idx)

        indices_mask = spans_to_indices(all_spans_tensor, self.max_span_length).long()
        if not self.use_all_subtoken_spans:
            # uses only the spans of all_spans

            span_mask_subtoks = torch.zeros((span_begin.size(0), span_begin.size(1), self.max_span_length),
                                            device=settings.device, dtype=torch.bool)
            span_mask_subtoks[0, :, :].view(-1)[indices_mask[0]] = True
            assert span_mask_subtoks.sum(dim=(-1, -2)).int().item() == all_spans_tensor.shape[1]
            span_mask[0] = span_mask[0] & span_mask_subtoks[0]

        for curr_batch in range(span_begin.size(0)):
            # span_mask[curr_batch, :, :].view(-1)[indices_mask[curr_batch]] = True

            # todo: move this code to collate, find a better alternative for this if: basically now is executed
            #  only when a single coreference is enabled without any linking component (ergo no entity_dictionary)
            if self.entity_dictionary is not None and (self.coref_linker_task.enabled or self.linker_task.enabled):
                tot_candidates = self.max_span_length * sequence_lengths[curr_batch].item()
                expanded_candidates = torch.zeros(tot_candidates, metadata['linker']['candidates'].shape[-1],
                                                  dtype=torch.long, device=settings.device)
                expanded_candidates[:, :] = self.entity_dictionary.lookup("###UNKNOWN###")

                expanded_candidates[indices_mask[curr_batch], :] = metadata['linker']['candidates'][curr_batch]

                expanded_cand_lengths = torch.zeros(tot_candidates, dtype=torch.long, device=settings.device)
                expanded_cand_lengths[indices_mask[curr_batch]] = metadata['linker']['candidate_lengths'][curr_batch]

                expanded_cand_targets = torch.zeros(tot_candidates, metadata['linker']['targets'].shape[-1],
                                                    dtype=torch.float, device=settings.device)
                expanded_cand_targets[indices_mask[curr_batch], :] = metadata['linker']['targets'][curr_batch]

                # now applying span_mask calculated above to the candidates, first some asserts to check that the
                # dimensions match between candidates and spans
                # assert expanded_candidates
                m1 = span_mask[0].shape[0]
                m2 = span_mask[0].shape[1]
                assert m1 * m2 == tot_candidates
                span_mask_flat = span_mask.view(-1)
                expanded_candidates = expanded_candidates[span_mask_flat, :]
                expanded_cand_lengths = expanded_cand_lengths[span_mask_flat]
                expanded_cand_targets = expanded_cand_targets[span_mask_flat, :]
                metadata['linker']['candidates'] = expanded_candidates.unsqueeze(0)
                metadata['linker']['candidate_lengths'] = expanded_cand_lengths.unsqueeze(0)
                metadata['linker']['targets'] = expanded_cand_targets.unsqueeze(0)

                # END TODO: move this code to collate!!

                # TODO CRITICAL!!!: also adapts the end-to-end linker candidate list, since the width of some of the passed
                #  spans can be > than self.max_span_length, there should be also a mapping between this candidate list
                #  and the absolute position given the self.max_span>length

                # TODO: update metadata['linker']['all_spans']

        # for loop for assertion purposes only, slow this is why it is commented
        # for curr_span_idx in all_spans_tensor[0]:
        #     span_width = (curr_span_idx[1] - curr_span_idx[0]).item()
        #     assert span_mask[0, curr_span_idx[0], span_width] == 1.0

        # extract span embeddings
        # only with batch size 1 for now
        # assert span_begin.shape[0] == 1
        span_begin = span_begin[0][span_mask[0]].unsqueeze(0)
        span_end = span_end[0][span_mask[0]].unsqueeze(0)
        #
        span_vecs, candidate_width_idx = \
            self.span_extractor(embeddings.unsqueeze(0), span_begin, span_end, self.max_span_length, span_mask)
        # span_vecs = self.span_extractor(hidden, span_begin, span_end, max_span_length)
        # span_vecs.shape --> ??

        all_spans = {
            # 'span_vecs': span_vecs,
            # 'span_begin': span_begin,
            # 'span_end': span_end,
            # 'span_mask': span_mask
            'cand_span_vecs': span_vecs,
            'cand_span_begin': span_begin,
            'cand_span_end': span_end,
            'cand_span_mask': span_mask,
            'cand_width_idx': candidate_width_idx
        }

        # gold_spans_tensors = metadata.get('gold_spans_tensors')
        obj_pruner, all_spans, filtered_spans = self.span_pruner(all_spans,
                                                                 token_lengths,
                                                                 metadata.get('gold_spans_lengths'),
                                                                 metadata.get('gold_spans_tensors'),
                                                                 doc_id=metadata.get('identifiers'),
                                                                 api_call=metadata.get('api_call'),
                                                                 max_span_length=self.max_span_length,
                                                                 emb_span_width_prior=self.span_extractor.embed,
                                                                 predict=not self.training)
        pred_spans = filtered_spans['pruned_spans']
        # gold_spans = metadata.get('gold_spans')
        gold_spans = filtered_spans['gold_spans']

        # in the original pred_spans:
        # in the original gold_spans:

        if self.debug_memory:
            print("(pruner)", torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## spanprop (no extra labels)
        if self.span_prop is not None:
            all_spans, filtered_spans = self.span_prop(
                all_spans,
                filtered_spans,
                sequence_lengths
            )

        ## coref
        if self.coref_task.enabled:
            coref_all, coref_filtered, coref_scores = self.coref_scorer(
                all_spans,
                filtered_spans,
                # sequence_lengths,
                metadata.get('gold_spans_tensors'),
                max_span_length=self.max_span_length,
                gold_spans_lengths=metadata.get('gold_spans_lengths')
            )
        else:
            coref_scores = None

        if self.coref_linker_task.enabled:
            # kzaporoj - for now just like this, then can add different graph propagation for coref+linker configuration
            coref_all, coref_filtered, linker_coref_scores = self.coref_linker_scorer(
                all_spans,
                filtered_spans,
                sequence_lengths,
                linker=metadata['linker'])
        else:
            coref_all = all_spans
            coref_filtered = filtered_spans
            linker_coref_scores = None
            coref_targets = None

        if not self.rel_after_coref:
            coref_all = all_spans
            coref_filtered = filtered_spans

        if self.debug_memory:
            print("(coref) ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## relations
        # if self.relation_task.enabled:
        #     relation_all, relation_filtered, relation_scores = self.rel_scorer(
        #         coref_all,
        #         coref_filtered,
        #         sequence_lengths)
        #
        # else:
        #     relation_all = coref_all
        #     relation_filtered = coref_filtered
        #     relation_scores = None
        #     relation_targets = None

        if self.debug_memory:
            print("(rels)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # LOSS FUNCTIONS

        ## ner
        # ner_obj, output['tags'] = self.ner_task(
        #     relation_all,
        #     sequence_lengths,
        #     metadata.get('gold_tags_indices'),
        #     api_call=metadata.get('api_call')
        # )
        #
        # ner_spans = [list(set([(begin, end - 1) for begin, end, _ in spans])) for spans in
        #              output['tags']['pred']]  # TODO: settle on include spans

        coref_obj, output['coref'] = self.coref_task(
            coref_scores,
            gold_m2i=metadata.get('gold_m2i'),
            pred_spans=pred_spans,
            gold_spans=gold_spans,
            predict=not self.training,
            pruner_spans=coref_filtered['enabled_spans'],
            span_lengths=filtered_spans['span_lengths'],
            ner_spans=None
            # ner_spans=ner_spans
        )

        # "atadura con alambre" to be able to produce the matrix print for debugging with Johannes
        filtered_spans['content'] = metadata['content']
        filtered_spans['begin_token'] = metadata['begin_token']
        filtered_spans['subtoken_map'] = metadata['subtoken_map']
        filtered_spans['end_token'] = metadata['end_token']
        filtered_spans['doc_id'] = metadata['identifiers']
        # filtered_spans['debug_path'] = config

        ## linker+coref
        linker_coref_obj, output['links'], output_coref = \
            self.coref_linker_task(
                scores=linker_coref_scores,
                gold_m2i=metadata.get('gold_m2i'),
                filtered_spans=filtered_spans,
                gold_spans=gold_spans,
                linker=metadata['linker'],
                # predict=True,
                predict=not self.training,
                pruner_spans=coref_filtered['enabled_spans'],
                # ner_spans=ner_spans,
                ner_spans=None,
                api_call=metadata.get('api_call'),
                only_loss=only_loss
            )  # TODO: candidate links

        # if not self.coref_task.enabled:
        if self.coref_linker_task.enabled:
            # if the coreflinker task is enabled, then overrides the corefs with the ones calculated by coreflinker
            output['coref'] = output_coref

        ## relations
        # rel_obj, output['rels'] = self.relation_task(
        #     relation_filtered,
        #     relation_scores,
        #     relations,
        #     output['coref'],
        #     predict=not self.training
        # )

        ## linker
        if self.linker_task.enabled:
            linker_obj, output_links, output['coref'] = self.linker_task(
                # relation_all,
                coref_all,
                metadata['linker'],
                filtered_spans,
                gold_m2i=metadata.get('gold_m2i'),
                gold_spans=gold_spans,
                coref_pred=output['coref'] if
                (self.coref_task.enabled and not self.training and not only_loss) else None,
                predict=not self.training,
                only_loss=only_loss
            )
            output['links'] = output_links

            # if not self.coref_task.enabled:
            #     # if the coref task is not enabled, then takes the coreference given by linking (mentions pointing
            #     # to the same link are considered clustered).
            #     output['coref'] = tmp_out_coref
        else:
            linker_obj, _ = self.linker_task(
                coref_all,
                # relation_all,
                metadata['linker']
            )

        for m in metrics:
            if m.task in output:
                m.update2(output[m.task], metadata)

        if self.debug_memory:
            print("(loss)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # return obj_pruner + coref_obj + ner_obj + rel_obj + linker_obj + linker_coref_obj, output
        return obj_pruner + coref_obj + linker_obj + linker_coref_obj, output

    def predict(self, inputs, metadata, metrics=[], output_config=None):
        loss, output = self.forward(inputs, metadata, metrics)
        return loss, self.decode(metadata, output, output_config)

    def create_metrics(self):
        return self.coref_task.create_metrics() + \
               self.linker_task.create_metrics() + self.coref_linker_task.create_metrics()
        # self.ner_task.create_metrics() + \
        # self.relation_task.create_metrics() \
        # + self.coref_linker_mtt_task.create_metrics()

    def write_model(self, filename):
        print("write model:", filename)
        mydict = {}
        for k, v in self.state_dict().items():
            if k.startswith('word_embeddings'):
                print("skip:", k)
                continue
            else:
                mydict[k] = v
        torch.save(mydict, filename)

    def load_model(self, filename, to_cpu=False, load_word_embeddings=True):
        print('to_cpu IN LOAD_MODEL: ', to_cpu)
        if to_cpu:
            partial = torch.load(filename, map_location=torch.device('cpu'))
        else:
            # partial = torch.load(filename)
            partial = torch.load(filename, map_location=torch.device(settings.device))

        if not load_word_embeddings:
            keys_to_remove = []
            for curr_key in partial.keys():
                if 'embedder.word_embedder' in curr_key:
                    keys_to_remove.append(curr_key)
                if 'entity_embedder.embed.weight' in curr_key:
                    keys_to_remove.append(curr_key)
            for curr_key in keys_to_remove:
                del partial[curr_key]
        # update because word_embed is missing
        state = self.state_dict()
        state.update(partial)
        self.load_state_dict(state)

    def pass_from_subtokens_to_tokens(self, outputs, subtoken_to_token_map):
        # TODO! not only pass from BERT subtokens to tokens, but also de-duplicate (ex: delete duplicate spans in
        #  clusters). This can happen because two different BERT subtokens can map to the same token.
        new_coref = []
        new_coref_pointers = []
        new_coref_scores = []
        new_links_scores = []
        new_links_pred = []
        new_links_gold = []

        def get_coref_pointer(coref_connection_pointer, coref_connection_type, t_map: Dict):
            if coref_connection_pointer is None:
                return None

            if coref_connection_type != 'link':
                return t_map[coref_connection_pointer[0]], t_map[coref_connection_pointer[1]]
            else:
                # should be string (i.e., the entity link)
                assert isinstance(coref_connection_pointer, str)
                return coref_connection_pointer

        def get_coref_pointer_dict_entry(span, v, t_map: Dict):
            to_ret = dict()
            to_ret['coref_connection_type'] = v['coref_connection_type']
            to_ret['coref_connection_pointer'] = get_coref_pointer(v['coref_connection_pointer'],
                                                                   v['coref_connection_type'], t_map)
            # (t_map[v['coref_connection_pointer'][0]],
            #  t_map[v['coref_connection_pointer'][1]])
            # if v['coref_connection_pointer'] is not None else None,
            if 'coref_connection_score' in v:
                to_ret['coref_connection_score'] = v['coref_connection_score']

            # }
            return to_ret

        for coref, coref_pointers, coref_scores, \
            links_scores, links_gold, links_pred, \
            t_map in zip(outputs['coref']['pred'], outputs['coref']['pred_pointers'], outputs['coref']['scores'],
                         outputs['links']['scores'], outputs['links']['gold'], outputs['links']['pred'],
                         subtoken_to_token_map):
            #####
            # print('an iteration over pass_from_subtokens_to_tokens')
            tok_coref = [[(t_map[span[0]], t_map[span[1]]) for span in cluster] for cluster in coref]
            new_coref.append(tok_coref)
            if coref_pointers is not None:
                tok_coref_pointers = {(t_map[span[0]], t_map[span[1]]):
                                      get_coref_pointer_dict_entry(span, v, t_map)
                                  # {'coref_connection_type': v['coref_connection_type'],
                                  #  'coref_connection_pointer':
                                  #      get_coref_pointer(v['coref_connection_pointer'], v['coref_connection_type'],
                                  #                        t_map),
                                  #  'coref_connection_score': v['coref_connection_score']
                                  #  }
                                  for span, v in
                                  coref_pointers.items()}
            else:
                tok_coref_pointers = dict()
            new_coref_pointers.append(tok_coref_pointers)

            if coref_scores is not None:
                tok_coref_scores = {(t_map[span[0]], t_map[span[1]]): [{'span': (t_map[v['span'][0]], t_map[v['span'][1]]),
                                                                    'score': v['score']}
                                                                   for v in values] for span, values in
                                coref_scores.items()}
            else:
                tok_coref_scores = dict()
            new_coref_scores.append(tok_coref_scores)

            tok_links_scores = [((t_map[span[0]], t_map[span[1]]), links, scores) for span, links, scores in
                                links_scores]
            new_links_scores.append(tok_links_scores)

            tok_links_pred = [(t_map[link_pred[0]], t_map[link_pred[1]], link_pred[2]) for link_pred in links_pred]
            new_links_pred.append(tok_links_pred)

            tok_links_gold = [(t_map[link_gold[0]], t_map[link_gold[1]], link_gold[2]) for link_gold in links_gold]
            new_links_gold.append(tok_links_gold)

        outputs['coref']['pred'] = new_coref
        outputs['coref']['pred_pointers'] = new_coref_pointers
        outputs['coref']['scores'] = new_coref_scores
        outputs['links']['scores'] = new_links_scores
        outputs['links']['gold'] = new_links_gold
        outputs['links']['pred'] = new_links_pred

    # coref = {list} <class 'list'>: [[(92, 93), (93, 94)]]
    # coref_pointers = {dict} <class 'dict'>: {(9, 10): {'coref_connection_type': 'root', 'coref_connection_pointer': (9, 10), 'coref_connection_score': -1.5209635496139526}, (11, 12): {'coref_connection_type': 'link', 'coref_connection_pointer': None, 'coref_connection_score': -1.4718
    # coref_scores = {dict} <class 'dict'>: {(92, 93): [{'span': (9, 10), 'score': -2.634833335876465}, {'span': (11, 12), 'score': -2.6334939002990723}, {'span': (67, 67), 'score': -2.64146089553833}, {'span': (71, 72), 'score': -2.741600751876831}, {'span': (74, 75), 'score': -2.74
    # outputs['links']['scores'] = {list} <class 'list'>: [((92, 93), ['Defensive_end', 'List_of_United_States_senators_from_Delaware', 'Germany', 'German_language', 'Delaware', 'GfK_Entertainment_charts', '.de', 'De_(Chinese)', 'Delaware_Democratic_Party', 'Haplogroup_DE', 'Nazi_Germany', 'De_(C
    # links_gold = {list} <class 'list'>: [(7, 12, None), (51, 53, None), (55, 59, None), (61, 61, 'Berlin'), (63, 67, None), (68, 72, 'wikidata:Q53666408'), (76, 76, 'Berlin')]
    # links_pred = {list} <class 'list'>: [(92, 93, '.de'), (93, 94, '.de')]
    # subtoken_to_tok_map = {list} <class 'list'>: [0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 37, 37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 45, 46, 47, 48, 49, 50, 51

    def decode(self, metadata, outputs, output_config, api_call=False):
        subtoken_to_token_map = metadata['subtoken_map']
        self.pass_from_subtokens_to_tokens(outputs, subtoken_to_token_map)

        predictions = []

        links_gold_batch = outputs['links']['gold']
        identifiers = metadata['identifiers']
        tags_batch = metadata['tags']

        idx = 0
        for identifier, tags, content, begin_token, end_token, coref, coref_pointers, coref_scores, \
            links_scores, links_gold, links_pred in zip(
            identifiers, tags_batch, metadata['content'], metadata['begin_token'], metadata['end_token'],
            outputs['coref']['pred'], outputs['coref']['pred_pointers'],
            outputs['coref']['scores'],
            outputs['links']['scores'], links_gold_batch, outputs['links']['pred']):
            ####
            predictions.append(
                convert_to_json(identifier, tags, content, begin_token, end_token, [], coref,
                                coref_pointers, coref_scores, [],
                                [], links_scores, links_gold, links_pred,
                                singletons=self.coref_task.singletons, output_config=output_config))
            idx += 1

        return predictions

# extra old code:
#             # span_widths = (all_spans_tensor[curr_batch, :, 1] - all_spans_tensor[curr_batch, :, 0])
#             # filter_span_width = (span_widths < self.max_span_length)
#
#             # reduces the all_spans to only those whose width is <= self.max_span_length
#             # all_spans_tensor_filtered = all_spans_tensor[curr_batch, filter_span_width, :]
#             # indices_mask = (all_spans_tensor[curr_batch, :, 0] + (span_widths * span_begin.shape[1])).long()
#             # span_widths_filtered = span_widths[filter_span_width]
#             # indices_mask = (all_spans_tensor_filtered[:, 0] + (span_widths_filtered * span_begin.shape[1])).long()
#             # span_mask[curr_batch, :, :].view(-1)[indices_mask] = 1.0
