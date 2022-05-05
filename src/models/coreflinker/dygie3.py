import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import settings
from cpn.builder import convert_to_json
from datass.collate import collate_character
# from models.dygie import collate_dygie
from models.coreflinker.attention import ModulePlainAttention
from models.coreflinker.attprop import ModuleAttentionProp
from models.coreflinker.corefbasic import ModuleCorefBasic
from models.coreflinker.coreflinker_mtt_prop import ModuleCorefLinkerMTTPropE2E
from models.coreflinker.coreflinker_prop import ModuleLinkerCorefProp, ModuleCorefLinkerPropE2E
from models.coreflinker.corefprop import ModuleCorefProp
from models.coreflinker.corefprop2 import ModuleCorefProp2
from models.coreflinker.pruner import MentionPruner, MentionPrunerGold
from models.coreflinker.relbasic import ModuleRelBasic
from models.coreflinker.relprop import ModuleRelProp, ModuleRelPropX
from models.coreflinker.relprop1 import ModuleRelProp1
from models.coreflinker.relprop2 import ModuleRelProp2
from models.coreflinker.relsimple import ModuleRelSimple
from modules.misc.misc import batched_index_select
from modules.ner.spanner import TaskSpan1x, create_all_spans, create_span_extractor
from modules.spirit import TextEmbedder, Seq2Seq
from modules.tasks.coref import LossCoref, LossBidirectionalCoref
from modules.tasks.coreflinker import CorefLinkerLoss
from modules.tasks.coreflinker_edmonds_softmax import LossCorefLinkerESM
from modules.tasks.coreflinker_mtt import LossCorefLinkerMTT
from modules.tasks.linker import LinkerNone, LossLinker, collate_candidates_in_pytorch, \
    LossLinkerE2E, collate_targets, collate_tot_cand_lengths
from modules.tasks.linker import collate_spans
from modules.tasks.relations import create_task_relations
from modules.utils.misc import SpanPairs, inspect


# from allennlp.nn.util import batched_index_select


def collate_dygie(model, batch, device, collate_api=False):
    """

    :param model:
    :param batch:
    :param device:
    :param collate_api: if in True, means that the input comes from a client, possibly as a free text
    (i.e., no gold mentions, relations, concepts, spans, etc.). If in False (default), the input comes for training
    or evaluating using internal function located in traintool.train for instance.
    :return:
    """
    # print('\ncollate_dygie for following doc ids: ', [b['id'] for b in batch])
    batch.sort(key=lambda x: x['xxx']['tokens'].size()[0], reverse=True)
    sequence_lengths = torch.LongTensor([x['xxx']['tokens'].size()[0] for x in batch])
    # TODO: move this to TextFieldEmbedderCharacters
    if model.embedder.do_char_embedding:
        characters = collate_character([x['xxx']['characters'] for x in batch], 50,
                                       model.embedder.char_embedder.padding,
                                       min_word_len=model.embedder.char_embedder.min_word_length)
    else:
        characters = None

    tokens = rnn_utils.pad_sequence([x['xxx']['tokens'] for x in batch], batch_first=True)
    last_idx = max([len(x['xxx']['tokens']) for x in batch]) - 1
    indices = rnn_utils.pad_sequence([x['xxx']['tokens-indices'] for x in batch], batch_first=True,
                                     padding_value=last_idx)

    inputs = {
        'tokens': tokens.to(device),
        'characters': characters.to(device) if characters is not None else None,
        'sequence_lengths': sequence_lengths.to(device),
        'token_indices': indices.to(device),
        'text': [b['xxx']['text'] for b in batch]
    }
    # if not collate_api:
    gold_spans = [[(m[0], m[1]) for m in x['spans']] for x in batch]
    # end if not collate_api

    # if not collate_api:
    if 'gold_clusters' in batch[0]:
        gold_clusters = [x['gold_clusters'] for x in batch]
    else:
        # TODO: move to cpn utility .py (or remove)
        gold_clusters = []
        for spans, m2c in zip(gold_spans, [x['mention2concept'] for x in batch]):
            clusters = [list() for _ in range(m2c[0])]
            for mention, concept in zip(m2c[3], m2c[2]):
                clusters[concept].append(spans[mention])
            gold_clusters.append(clusters)
    # end if not collate_api:

    metadata = {
        'tokens': [x['xxx']['text'] for x in batch],
        'content': [x['content'] for x in batch],
        'begin': [x['begin'] for x in batch],
        'end': [x['end'] for x in batch]
    }

    # if not collate_api:
    metadata['identifiers'] = [x['id'] for x in batch]
    metadata['tags'] = [x['metadata_tags'] for x in batch]
    # end if not collate_api:

    relations = None
    # if not collate_api:
    metadata['gold_tags_indices'] = [x['gold_tags_indices'] for x in batch]
    metadata['gold_spans'] = gold_spans
    metadata['gold_spans_lengths'] = (torch.LongTensor([len(curr_spans) for curr_spans in gold_spans])).to(
        device=settings.device)
    metadata['gold_m2i'] = [x['clusters'] for x in batch]

    relations = {
        'gold_spans': gold_spans,
        'gold_m2i': [x['clusters'] for x in batch],
        'gold_clusters2': gold_clusters
    }

    if 'relations' in batch[0]:
        # old: remove the dimension
        relations['gold_relations'] = [x['relations'][1] for x in batch]
        relations['num_concepts'] = [x['relations'][0][0] for x in batch]
    else:
        relations['gold_relations'] = [x['relations2'] for x in batch]
        relations['num_concepts'] = [x['num_concepts'] for x in batch]

    spans_tensors = collate_spans(gold_spans)
    metadata['gold_spans_tensors'] = spans_tensors.to(device=settings.device)

    linker = {}
    if (model.linker_task.enabled or model.coref_linker_task.enabled) and 'linker_candidates' in batch[0]:
        # or model.coref_linker_mtt_task.enabled)
        candidates, candidate_lengths = collate_candidates_in_pytorch([x['linker_candidates'] for x in batch],
                                                                      unknown_id=model.entity_dictionary.lookup(
                                                                          '###UNKNOWN###'))
        linker['candidates'] = candidates
        linker['candidate_lengths'] = candidate_lengths
        # if not collate_api:
        linker['targets'] = collate_targets([x['linker_targets'] for x in batch], candidates.size(2))

        linker['total_cand_lengths_in_gold_mentions'] = \
            collate_tot_cand_lengths([torch.tensor(x['total_cand_lengths_in_gold_mentions'], dtype=torch.int32)
                                      for x in batch])

        linker['gold'] = [x['linker_gold'] for x in batch]
        # end if not collate_api:

    metadata['linker'] = linker
    metadata['api_call'] = collate_api

    return {
        'inputs': inputs,
        'relations': relations,
        'metadata': metadata
    }


def create_spanprop(model, config):
    if 'spanprop' in config:
        sp_type = config['spanprop']['type']

        if sp_type == 'attprop':
            return ModuleAttentionProp(model.span_extractor.dim_output,
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
    cp_type = config['coreflinker']['coreflinker_prop']['type']

    if cp_type == 'none':
        return None
    elif cp_type == 'default':
        coreflinker_type = config['coreflinker']['type']
        if model.end_to_end_mentions:
            if coreflinker_type == 'coreflinker':
                return ModuleCorefLinkerPropE2E(model.span_extractor.dim_output,
                                                model.span_pruner.scorer,
                                                model.span_pair_generator, config['coreflinker'], dictionaries)
            elif coreflinker_type == 'coreflinker_mtt':
                return ModuleCorefLinkerMTTPropE2E(model.span_extractor.dim_output,
                                                   model.span_pruner.scorer,
                                                   model.span_pair_generator, config['coreflinker'], dictionaries)
            elif coreflinker_type == 'coreflinker_esm':
                return ModuleCorefLinkerPropE2E(model.span_extractor.dim_output,
                                                model.span_pruner.scorer,
                                                model.span_pair_generator, config['coreflinker'], dictionaries)
        else:
            if coreflinker_type == 'coreflinker':
                return ModuleLinkerCorefProp(model.span_extractor.dim_output,
                                             model.span_pruner.scorer,
                                             model.span_pair_generator, config['coreflinker'], dictionaries)
            elif coreflinker_type == 'coreflinker_mtt':
                return ModuleCorefLinkerMTTPropE2E(model.span_extractor.dim_output,
                                                   model.span_pruner.scorer,
                                                   model.span_pair_generator, config['coreflinker'], dictionaries)
            elif coreflinker_type == 'coreflinker_esm':
                return ModuleLinkerCorefProp(model.span_extractor.dim_output,
                                             model.span_pruner.scorer,
                                             model.span_pair_generator, config['coreflinker'], dictionaries)
    else:
        raise BaseException("no such linkercoref prop:", cp_type)
    raise BaseException("no such coreflinker found (in coreflinker_prop):", config['coreflinker'])


def create_coreflinker_loss(model, config):
    coreflinker_type = config['coreflinker']['type']
    if model.coref_linker_scorer.enabled:
        if coreflinker_type == 'coreflinker':
            return CorefLinkerLoss('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                   config['coreflinker'], model.end_to_end_mentions)
        elif coreflinker_type == 'coreflinker_mtt':
            return LossCorefLinkerMTT('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                      config['coreflinker'], model.end_to_end_mentions)
        elif coreflinker_type == 'coreflinker_esm':
            return LossCorefLinkerESM('links', 'coref', model.coref_linker_scorer.entity_embedder.dictionary,
                                      config['coreflinker'], model.end_to_end_mentions)
    else:
        return CorefLinkerLoss('links', 'coref', None, config['coreflinker'], model.end_to_end_mentions)

    raise BaseException("no such coreflinker found (in create_coreflinker_loss):", config['coreflinker'])


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


class MyDygie3(nn.Module):

    def __init__(self, dictionaries, config):
        super(MyDygie3, self).__init__()
        self.random_embed_dim = config['random_embed_dim']
        self.max_span_length = config['max_span_length']
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.rel_after_coref = config['rel_after_coref']
        # self.spanbert_input = config['spanbert_input']

        self.load_doc_level_candidates = \
            (config['coreflinker']['enabled'] and config['coreflinker']['doc_level_candidates']) or \
            (config['linker']['enabled'] and config['linker']['doc_level_candidates'])

        self.debug_memory = False
        self.debug_tensors = False

        # whether take gold mentions or use the pruner
        self.end_to_end_mentions = config['end_to_end_mentions']
        self.embedder = TextEmbedder(dictionaries, config['text_embedder'])
        self.entity_dictionary = dictionaries['entities']

        self.emb_dropout = nn.Dropout(config['lexical_dropout'])
        self.seq2seq = Seq2Seq(self.embedder.dim_output + self.random_embed_dim, config['seq2seq'])

        self.span_extractor = create_span_extractor(self.seq2seq.dim_output, self.max_span_length,
                                                    config['span-extractor'])

        if self.end_to_end_mentions:
            self.span_pruner = MentionPruner(self.span_extractor.dim_output, self.max_span_length, config['pruner'])
        else:
            self.span_pruner = MentionPrunerGold(self.max_span_length, config['pruner'])

        self.span_pair_generator = SpanPairs(self.span_extractor.dim_output, config['span-pairs'])

        self.span_prop = create_spanprop(self, config)

        self.coref_scorer = create_corefprop(self, config['coref'])

        self.relation_labels = dictionaries['relations'].tolist()
        self.rel_scorer = create_relprop(self, config)

        if not config['coref']['bidirectional']:
            self.coref_task = LossCoref('coref', config['coref'])
        else:
            self.coref_task = LossBidirectionalCoref('coref', config['coref'])

        # kzaporoj - here add linkercoref joint
        self.coref_linker_scorer = create_coreflinker_prop(self, config, dictionaries)
        self.coref_linker_task = create_coreflinker_loss(self, config)

        self.ner_task = TaskSpan1x('tags', self.span_extractor.dim_output, dictionaries['tags'], config['ner'])
        self.relation_task = create_task_relations('rels', config['relations'], self.relation_labels)

        if 'linker' in config and config['linker']['enabled']:
            if self.end_to_end_mentions:
                self.linker_task = LossLinkerE2E('links', self.span_extractor.dim_output, dictionaries,
                                                 config['linker'], self.max_span_length)
            else:
                self.linker_task = LossLinker('links', self.span_extractor.dim_output, dictionaries, config['linker'])
        else:
            self.linker_task = LinkerNone()

        if not self.span_pruner.sort_after_pruning and self.pairs.requires_sorted_spans:
            raise BaseException("ERROR: spans MUST be sorted")

    def collate_func(self, datasets, device):
        return lambda x: collate_dygie(self, x, device)

    def end_epoch(self, dataset_name):
        self.span_pruner.end_epoch(dataset_name)

    def forward(self, inputs, relations, metadata, metrics=[]):
        output = {}

        sequence_lengths = inputs['sequence_lengths']

        if self.debug_memory or self.debug_tensors:
            print("START", metadata['identifiers'][0], sequence_lengths)
            print("(none)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # MODEL MODULES
        embeddings = self.embedder(inputs)

        embeddings = self.emb_dropout(embeddings)

        if self.random_embed_dim > 0:
            rand_embedding = torch.FloatTensor(embeddings.size(0), embeddings.size(1), self.random_embed_dim).to(
                embeddings.device).normal_(std=4.0)
            rand_embedding = batched_index_select(rand_embedding, inputs['token_indices'])
            embeddings = torch.cat((embeddings, rand_embedding), -1)

        if self.debug_tensors:
            inspect('embeddings', embeddings[0, :, :])

        hidden = self.seq2seq(embeddings, sequence_lengths, inputs['token_indices']).contiguous()

        if self.debug_tensors:
            inspect('hidden', hidden[0, :, :])

        # create span
        span_begin, span_end = create_all_spans(hidden.size(0), hidden.size(1), self.max_span_length)
        # span_begin.shape -->  1 x 69 x 5     ; span_end.shape --> 1 x 69 x 5
        # kzaporoj
        if settings.device == 'cuda':
            span_begin, span_end = span_begin.cuda(), span_end.cuda()
        else:
            span_begin, span_end = span_begin.cpu(), span_end.cpu()

        span_mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float()  # span_mask.shape --> 1 x 69 x 5

        # extract span embeddings
        span_vecs = self.span_extractor(hidden, span_begin, span_end, self.max_span_length)

        all_spans = {
            'span_vecs': span_vecs,
            'span_begin': span_begin,
            'span_end': span_end,
            'span_mask': span_mask
        }

        obj_pruner, all_spans, filtered_spans = self.span_pruner(all_spans, metadata.get('gold_spans'),
                                                                 sequence_lengths,
                                                                 metadata.get('gold_spans_lengths'),
                                                                 metadata.get('gold_spans_tensors'),
                                                                 doc_id=metadata.get('identifiers'),
                                                                 api_call=metadata.get('api_call'))
        # span_lengths = filtered_spans['span_lengths']
        pred_spans = filtered_spans['spans']
        gold_spans = metadata.get('gold_spans')

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
        if self.relation_task.enabled:
            relation_all, relation_filtered, relation_scores = self.rel_scorer(
                coref_all,
                coref_filtered,
                sequence_lengths)

        else:
            relation_all = coref_all
            relation_filtered = coref_filtered
            relation_scores = None
            relation_targets = None

        if self.debug_memory:
            print("(rels)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # LOSS FUNCTIONS

        ## ner
        ner_obj, output['tags'] = self.ner_task(
            relation_all,
            sequence_lengths,
            metadata.get('gold_tags_indices'),
            api_call=metadata.get('api_call')
        )

        ner_spans = [list(set([(begin, end - 1) for begin, end, _ in spans])) for spans in
                     output['tags']['pred']]  # TODO: settle on include spans

        coref_obj, output['coref'] = self.coref_task(
            coref_scores,
            gold_m2i=metadata.get('gold_m2i'),
            pred_spans=pred_spans,
            gold_spans=gold_spans,
            predict=True,
            pruner_spans=relation_filtered['enabled_spans'],
            ner_spans=ner_spans
        )

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
                pruner_spans=relation_filtered['enabled_spans'],
                ner_spans=ner_spans,
                api_call=metadata.get('api_call')
            )  # TODO: candidate links

        # if not self.coref_task.enabled:
        if self.coref_linker_task.enabled:
            # if the coreflinker task is enabled, then overrides the corefs with the ones calculated by coreflinker
            output['coref'] = output_coref

        ## relations
        rel_obj, output['rels'] = self.relation_task(
            relation_filtered,
            relation_scores,
            relations,
            output['coref'],
            predict=not self.training
        )

        ## linker
        if self.linker_task.enabled:
            linker_obj, output_links, tmp_out_coref = self.linker_task(
                relation_all,
                metadata['linker'],
                filtered_spans,
                gold_m2i=metadata.get('gold_m2i'),
                gold_spans=gold_spans
            )
            output['links'] = output_links

            if not self.coref_task.enabled:
                # if the coref task is not enabled, then takes the coreference given by linking (mentions pointing
                # to the same link are considered clustered).
                output['coref'] = tmp_out_coref
        else:
            linker_obj, _ = self.linker_task(
                relation_all,
                metadata['linker']
            )

        for m in metrics:
            if m.task in output:
                m.update2(output[m.task], metadata)

        if self.debug_memory:
            print("(loss)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        return obj_pruner + coref_obj + ner_obj + rel_obj + linker_obj + linker_coref_obj, output

    def predict(self, inputs, relations, metadata, metrics=[], output_config=None):
        loss, output = self.forward(inputs, relations, metadata, metrics)
        return loss, self.decode(metadata, output, output_config)

    def create_metrics(self):
        return self.coref_task.create_metrics() + self.ner_task.create_metrics() + self.relation_task.create_metrics() \
               + self.linker_task.create_metrics() + self.coref_linker_task.create_metrics()
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

    def decode(self, metadata, outputs, output_config, api_call=False):
        predictions = []

        links_gold_batch = outputs['links']['gold']
        identifiers = metadata['identifiers']
        tags_batch = metadata['tags']

        idx = 0
        for identifier, tags, content, begin, end, ner, coref, coref_pointers, coref_scores, \
            concept_rels, span_rels, links_scores, links_gold, \
            links_pred in zip(
            identifiers, tags_batch, metadata['content'], metadata['begin'], metadata['end'],
            outputs['tags']['pred'], outputs['coref']['pred'], outputs['coref']['pred_pointers'],
            outputs['coref']['scores'],
            outputs['rels']['pred'],
            outputs['rels']['span-rel-pred'], outputs['links']['scores'], links_gold_batch,
            outputs['links']['pred']):
            predictions.append(
                convert_to_json(identifier, tags, content, begin.tolist(), end.tolist(), ner, coref,
                                coref_pointers, coref_scores, concept_rels,
                                span_rels, links_scores, links_gold, links_pred,
                                singletons=self.coref_task.singletons, output_config=output_config))
            idx += 1

        return predictions
