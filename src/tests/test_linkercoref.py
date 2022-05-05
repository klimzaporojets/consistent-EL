import unittest

import torch

# the goal of this module is to add a bunch of tests with asserts in order to check that the decoding process is
# performed correctly. The most critical part of code that I think can potentially have bugs is
# modules.tasks.linkercoref.LossLinkerCoref#forward, which is going to be tested in this module.
import settings
from datass.dictionary import Dictionary
from modules.tasks.coreflinker import CorefLinkerLoss


class DwieLinkerCorefTest(unittest.TestCase):
    def test_scenario_001(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (5,5)
        - the nr of candidates is 2 for the rest of the mentions
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_001')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin'), dic.lookup('Berlin2')],
                                                 [dic.lookup('Ghent'), dic.lookup('Ghent2')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[1, 0],
                                           [0, 0],
                                           [0, 1],
                                           [0, 0],
                                           [0, 0]]])
        linker['candidate_lengths'] = torch.IntTensor([[3, 3, 3, 3, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[3, 3, 3, 3, 0]])

        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'NILL'), (3, 3, 'Berlin'), (4, 4, 'NILL')]]

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, False
                                            # config['linkercoref']
                                            )

        scores = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True, pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'NILL'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'NILL')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    def test_scenario_002(self):
        """
        - no_nil_in_targets in True
        - all mention have candidate links
        - the nr of candidates is 2 for the rest of the mentions
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_002')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin'), dic.lookup('Berlin2')],
                                                 [dic.lookup('Ghent'), dic.lookup('Ghent2')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent')]]])
        linker['targets'] = torch.Tensor([[[1, 0],
                                           [0, 0],
                                           [0, 1],
                                           [0, 0]]])
        linker['candidate_lengths'] = torch.IntTensor([[3, 3, 3, 3]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[3, 3, 3, 3]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'NILL'), (3, 3, 'Berlin'), (4, 4, 'NILL')]]

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, False
                                            # config['linkercoref']
                                            )

        scores = torch.Tensor([[[1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 1, 0, 0],  # Ghent
                                [0, 1, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 1, 0, 0]  # Ghent
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True, pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'NILL'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'NILL')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)]]

    def test_scenario_003(self):
        """
        - no_nil_in_targets in False
        - all mention have candidate links
        - the nr of candidates is 3 (including NILL) for the rest of the mentions
        - some of the valid candidates are NILL
        """

        print('EXECUTING DwieLinkerCorefTest.test_scenario_003')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': False,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('NILL'), dic.lookup('Berlin'), dic.lookup('Berlin2')],
                                                 [dic.lookup('Ghent'), dic.lookup('NILL'), dic.lookup('Ghent2')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'), dic.lookup('NILL')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 1, 0],
                                           [0, 1, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]])

        linker['candidate_lengths'] = torch.IntTensor([[3, 3, 3, 3]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[3, 3, 3, 3]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'NILL'), (3, 3, 'Berlin'), (4, 4, 'NILL')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            config_linkercoref, False
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0],  # Ghent
                                [0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 1, 0, 0]  # Ghent
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True, pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'NILL'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'NILL')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)]]

    def test_scenario_004(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (5,5)
        - the nr of candidates is 3 (including NILL) for the rest of the mentions
        """

        print('EXECUTING DwieLinkerCorefTest.test_scenario_004')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': False,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('NILL'), dic.lookup('Berlin'), dic.lookup('Berlin2')],
                                                 [dic.lookup('Ghent'), dic.lookup('NILL'), dic.lookup('Ghent2')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'), dic.lookup('NILL')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'), dic.lookup('NILL')]]])

        linker['targets'] = torch.Tensor([[[0, 1, 0],
                                           [0, 1, 0],
                                           [0, 1, 0],
                                           [0, 0, 1],
                                           [0, 0, 0]]])

        linker['candidate_lengths'] = torch.IntTensor([[3, 3, 3, 3, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[3, 3, 3, 3, 0]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'NILL'), (3, 3, 'Berlin'), (4, 4, 'NILL')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            config_linkercoref, False
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True, pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'NILL'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'NILL')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    # TESTS FOR DIFFERENT NUMBER OF CANDIDATES PER MENTION; mentions in same cluster with valid/invalid candidates
    def test_scenario_005(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (5,5)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_005')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()

        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin2'), dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 1],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]])
        # linker['candidate_lengths'] = torch.IntTensor([[5, 3, 3, 1, 0]])
        # linker['candidate_lengths_no_nill'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['candidate_lengths'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[5, 3, 3, 1, 0]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'Ghent'), (3, 3, 'Berlin'), (4, 4, 'Ghent')]]

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, False
                                            # config['linkercoref']
                                            )

        scores = torch.Tensor([[[0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

        # now making berlin to point to the link instead of the mention
        scores = torch.Tensor([[[0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    # TESTS FOR DIFFERENT NUMBER OF CANDIDATES PER MENTION; mentions in same cluster with valid/invalid candidates
    def test_scenario_006(self):
        """
        - no_nil_in_targets in False
        - mention without any candidate (ex: role) in span (5,5)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_006')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': False,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('NILL'), dic.lookup('Berlin2'),
                                                  dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('NILL'), dic.lookup('Berlin'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 0, 1],
                                           [0, 1, 0, 0, 0],
                                           [0, 0, 1, 0, 0],
                                           [1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]])

        linker['candidate_lengths'] = torch.IntTensor([[5, 3, 3, 1, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[5, 3, 3, 1, 0]])
        # linker['candidate_lengths_no_nill'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'Ghent'), (3, 3, 'Berlin'), (4, 4, 'Ghent')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, False
                                            # config['linkercoref']
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

        # now making point Berlin to the mention instead of the link
        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    # TESTS FOR DIFFERENT NUMBER OF CANDIDATES PER MENTION; mentions in same cluster with valid/invalid candidates
    def test_scenario_007(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (5,5)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        - mention with incorrect candidates occurs first => impossible to get cluster?
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_007')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': False,
                              'ignore_no_mention_chains':True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')

        linker = dict()

        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin2'), dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 1],
                                           [1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]])
        # linker['candidate_lengths'] = torch.IntTensor([[5, 1, 3, 3, 0]])
        linker['candidate_lengths'] = torch.IntTensor([[4, 0, 2, 2, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[5, 1, 3, 3, 0]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'Ghent'), (3, 3, 'Berlin'), (4, 4, 'Ghent')]]

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, False
                                            # config['linkercoref']
                                            )

        scores = torch.Tensor([[[0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]  # NILL
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        # THIS IS DOWNSIZE OF THE CURRENT APPROACH:
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'NILL'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'NILL')]

        # INSTEAD THIS ASSERTION SHOULD WORK: -> but not possible to refer to 'Ghent', since the first mention
        # doesn't have 'Ghent' in its candidates. This produces two wrong mentions!!!!
        # assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
        #                                                                  (2, 2, 'Ghent'),
        #                                                                  (3, 3, 'Berlin'),
        #                                                                  (4, 4, 'Ghent')]

        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    # TESTS FOR DIFFERENT NUMBER OF CANDIDATES PER MENTION; mentions in same cluster with valid/invalid candidates ;
    # filter_singletons_with_matrix in True; run in END-TO-END fashion
    def test_scenario_008(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (5,5)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        - some mentions are not valid entity mentions where filter_singletons_with_matrix enters in play (6,6)
            (ex: eleven) which is span but doesn't refer to any entity and is not valid entity mention
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_008')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': True,
                              'ignore_no_mention_chains': True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')
        dic.add('Eleven')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin2'),
                                                  dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'), dic.lookup('NILL'),
                                                  dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Eleven'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 1],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]])

        linker['candidate_lengths'] = torch.IntTensor([[4, 2, 2, 0, 0, 1]])

        # linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = None

        # linker['candidate_lengths_no_nill'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['gold_spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (2, 2, 'Ghent'), (3, 3, 'Berlin'), (4, 4, 'Ghent')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, True
                                            # config['linkercoref']
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]]
        filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
        filtered_spans['prune_indices'] = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

        # now making point Berlin to the mention instead of the link
        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (2, 2, 'Ghent'),
                                                                         (3, 3, 'Berlin'),
                                                                         (4, 4, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (3, 3)], [(2, 2), (4, 4)], [(5, 5)]]

    # TESTS FOR DIFFERENT NUMBER OF CANDIDATES PER MENTION; mentions in same cluster with valid/invalid candidates ;
    # filter_singletons_with_matrix in True; run in END-TO-END fashion
    def test_scenario_009(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (6,6)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        - some mentions are not valid entity mentions where filter_singletons_with_matrix enters in play (7,7)
            (ex: eleven) which is span but doesn't refer to any entity and is not valid entity mention
        - coreference from not-valid mention to other not valid mention: ignored (ignore_no_mention_chains in True)
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_009')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': True,
                              'ignore_no_mention_chains': True,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')
        dic.add('Eleven')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin2'),
                                                  dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('Eleven'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'), dic.lookup('NILL'),
                                                  dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Eleven'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 1],
                                           [0, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]])

        linker['candidate_lengths'] = torch.IntTensor([[4, 1, 2, 2, 0, 0, 1]])

        # linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = None

        # linker['candidate_lengths_no_nill'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['gold_spans'] = [[(1, 1), (3, 3), (4, 4), (5, 5), (6, 6)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (3, 3, 'Ghent'), (4, 4, 'Berlin'), (5, 5, 'Ghent')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, True
                                            # config['linkercoref']
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]]
        # filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
        filtered_spans['prune_indices'] = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # not a valid mention
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (3, 3, 'Ghent'),
                                                                         (4, 4, 'Berlin'),
                                                                         (5, 5, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (4, 4)], [(3, 3), (5, 5)], [(6, 6)]]

        # now making point Berlin to the mention instead of the link
        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # not a valid mention
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (3, 3, 'Ghent'),
                                                                         (4, 4, 'Berlin'),
                                                                         (5, 5, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (4, 4)], [(3, 3), (5, 5)], [(6, 6)]]

    def test_scenario_010(self):
        """
        - no_nil_in_targets in True
        - mention without any candidate (ex: role) in span (6,6)
        - variable nr of candidates for the rest of the mentions
        - some mentions in the same cluster have no correct candidates, while others do have correct ones
        - some mentions are not valid entity mentions where filter_singletons_with_matrix enters in play (7,7)
            (ex: eleven) which is span but doesn't refer to any entity and is not valid entity mention
        - coreference from not-valid mention to other not valid mention: added (ignore_no_mention_chains in False)
        """
        print('EXECUTING DwieLinkerCorefTest.test_scenario_010')
        config_linkercoref = {'enabled': True,
                              'weight': 1.0,
                              'filter_singletons_with_pruner': False,
                              'filter_singletons_with_ner': False,
                              'filter_singletons_with_matrix': True,
                              'ignore_no_mention_chains': False,
                              'end_to_end': False,
                              'no_nil_in_targets': True,
                              'doc_level_candidates': False}
        dic = Dictionary()
        dic.add('Berlin')
        dic.add('NILL')
        dic.add('Ghent')
        dic.add('Ghent2')
        dic.add('Ghent3')
        dic.add('Berlin2')
        dic.add('Berlin3')
        dic.add('Berlin4')
        dic.add('Eleven')

        linker = dict()
        linker['candidates'] = torch.IntTensor([[[dic.lookup('Berlin2'),
                                                  dic.lookup('Berlin4'),
                                                  dic.lookup('Berlin3'), dic.lookup('Berlin')],
                                                 [dic.lookup('Eleven'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Ghent2'), dic.lookup('Ghent'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Berlin2'), dic.lookup('Berlin'), dic.lookup('NILL'),
                                                  dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('NILL'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')],
                                                 [dic.lookup('Eleven'), dic.lookup('NILL'),
                                                  dic.lookup('NILL'), dic.lookup('NILL')]]])
        linker['targets'] = torch.Tensor([[[0, 0, 0, 1],
                                           [0, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]])

        linker['candidate_lengths'] = torch.IntTensor([[4, 1, 2, 2, 0, 0, 1]])

        # linker['total_cand_lengths_in_gold_mentions'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['total_cand_lengths_in_gold_mentions'] = None

        # linker['candidate_lengths_no_nill'] = torch.IntTensor([[4, 2, 2, 0, 0]])
        linker['gold_spans'] = [[(1, 1), (3, 3), (4, 4), (5, 5), (6, 6)]]
        linker['gold'] = [[(1, 1, 'Berlin'), (3, 3, 'Ghent'), (4, 4, 'Berlin'), (5, 5, 'Ghent')]]
        linker_coref_task = CorefLinkerLoss('links',
                                            'coref',
                                            dic,
                                            # self.linker_coref_scorer.entity_embedder.dictionary,
                                            config_linkercoref, True
                                            # config['linkercoref']
                                            )

        filtered_spans = {}
        filtered_spans['spans'] = [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]]
        # filtered_spans['reindex_wrt_gold'] = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
        filtered_spans['prune_indices'] = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.int32)

        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # not a valid mention
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (3, 3, 'Ghent'),
                                                                         (4, 4, 'Berlin'),
                                                                         (5, 5, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (4, 4)], [(3, 3), (5, 5)], [(2, 2), (7, 7)], [(6, 6)]]

        # now making point Berlin to the mention instead of the link
        scores = torch.Tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Berlin
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # not a valid mention
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Berlin
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Ghent
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # NILL
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # not a valid mention
                                ]])
        gold_m2i = [torch.IntTensor([0, 1, 0, 1, 2])]
        settings.device = 'cpu'
        output_loss, output_linking, output_coref = linker_coref_task(scores=scores, gold_m2i=gold_m2i,
                                                                      filtered_spans=filtered_spans,
                                                                      gold_spans=linker['gold_spans'],
                                                                      linker=linker, predict=True,
                                                                      pruner_spans=None,
                                                                      ner_spans=[[]])

        assert len(output_linking['pred']) == 1
        assert sorted(output_linking['pred'][0], key=lambda x: x[0]) == [(1, 1, 'Berlin'),
                                                                         (3, 3, 'Ghent'),
                                                                         (4, 4, 'Berlin'),
                                                                         (5, 5, 'Ghent')]
        assert len(output_coref['pred']) == 1
        assert output_coref['pred'][0] == [[(1, 1), (4, 4)], [(3, 3), (5, 5)], [(2, 2), (7, 7)], [(6, 6)]]
