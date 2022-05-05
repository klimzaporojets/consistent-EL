import torch

# the first goal of this module is to load the serialized coreflinker_mtt.py model along with input variables
# and re-run it to understand some errors (check conditions to trigger torch.save inside coreflinker_mtt.py for
# more details)
import settings
from modules.tasks.coreflinker_mtt import LossCorefLinkerMTT

if __name__ == '__main__':
    print('Start')
    print('loading model')
    loaded_model = torch.load('failed_model_scores_consolidated.bin')
    """
    {'scores': scores,
                            'gold_m2i': gold_m2i,
                            'filtered_spans': filtered_spans,
                            'gold_spans': gold_spans,
                            'linker': linker,
                            'predict': predict,
                            'pruner_spans': pruner_spans,
                            'ner_spans': ner_spans,
                            'api_call': api_call,
                            'model_state': self.state_dict()}
                            """
    scores = loaded_model['scores']
    gold_m2i = loaded_model['gold_m2i']
    filtered_spans = loaded_model['filtered_spans']
    gold_spans = loaded_model['gold_spans']
    linker = loaded_model['linker']
    predict = loaded_model['predict']
    pruner_spans = loaded_model['pruner_spans']
    ner_spans = loaded_model['ner_spans']
    api_call = loaded_model['api_call']
    model_state = loaded_model['model_state']
    config = dict()
    config['enabled'] = False
    config['weight'] = 1.0
    config['filter_singletons_with_pruner'] = True
    config['filter_singletons_with_ner'] = False
    config['float_precision'] = 'float64'
    config['multihead_nil'] = True

    # self.weight = config.get('weight', 1.0)
    # self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
    # self.filter_singletons_with_ner = config['filter_singletons_with_ner']
    # self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner
    # self.end_to_end = end_to_end
    # self.float_precision = config['float_precision']
    # self.multihead_nil = config['multihead_nil']
    model_mtt = LossCorefLinkerMTT(link_task='task', coref_task='task', entity_dictionary=None, config=config,
                                   end_to_end=True)
    model_mtt.enabled = True
    model_mtt.unknown_dict = 0
    settings.device = 'cpu'

    model_mtt(scores, gold_m2i, filtered_spans, gold_spans, linker, predict, pruner_spans, ner_spans, api_call)

    print('done')
    # def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
    #             predict=False, pruner_spans=None, ner_spans=None, api_call=False):
