from collections import Iterable

import torch
import torch.nn as nn
from torch.nn import init

from misc import settings


def create_pair_scorer(dim_input, dim_output, config, span_pair_generator):
    scorer_type = config.get('scorer_type', 'opt-ff-pairs')

    if scorer_type == 'ff-pairs':
        return FFpairs(dim_input, dim_output, config, span_pair_generator)
    elif scorer_type == 'opt-ff-pairs':
        return OptFFpairs(dim_input, dim_output, config, span_pair_generator)
    elif scorer_type == 'dot-pairs':
        return DotPairs(dim_input, dim_output, config, span_pair_generator)
    else:
        raise BaseException("no such pair scorer:", scorer_type)


class FFpairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(FFpairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.span_pair_generator = span_pair_generator
        self.scorer = nn.Sequential(
            nn.Linear(self.span_pair_generator.dim_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, dim_output)
        )

    def forward(self, span_vecs, span_begin, span_end):
        pairs = self.span_pair_generator(span_vecs, span_begin, span_end)
        return self.scorer(pairs)


class OptFFpairs(nn.Module):

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        if std is not None:
            init.normal_(linear.weight, std=std)
            if bias:
                init.zeros_(linear.bias)
            return linear

    def make_ffnn(self, feat_size, hidden_size, output_size, init_weights_std):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size, std=init_weights_std)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0], std=init_weights_std), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i], std=init_weights_std), nn.ReLU(),
                     self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size, std=init_weights_std))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(OptFFpairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4
        self.dropout = nn.Dropout(p=hidden_dp)
        self.init_weights_std = config['init_weights_std']
        self.components_ffnn_depth = config['components_ffnn_depth']
        self.scorers_ffnn_depth = config['scorers_ffnn_depth']

        self.span_pair_generator = span_pair_generator
        self.left = self.make_ffnn(feat_size=dim_input, hidden_size=[hidden_dim] * self.components_ffnn_depth,
                                   output_size=hidden_dim, init_weights_std=self.init_weights_std)
        self.right = self.make_ffnn(feat_size=dim_input, hidden_size=[hidden_dim] * self.components_ffnn_depth,
                                    output_size=hidden_dim, init_weights_std=self.init_weights_std)
        self.prod = self.make_ffnn(feat_size=dim_input, hidden_size=[hidden_dim] * self.components_ffnn_depth,
                                   output_size=hidden_dim, init_weights_std=self.init_weights_std)
        self.dist = self.make_ffnn(feat_size=span_pair_generator.dim_distance_embedding,
                                   hidden_size=[hidden_dim] * self.components_ffnn_depth,
                                   output_size=hidden_dim, init_weights_std=self.init_weights_std)

        self.scorer = self.make_ffnn(feat_size=hidden_dim, hidden_size=[hidden_dim] * self.scorers_ffnn_depth,
                                     output_size=dim_output, init_weights_std=self.init_weights_std)

    def forward(self, span_vecs, span_begin, span_end):
        p = self.span_pair_generator.get_product_embedding(span_vecs)
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)

        h = self.left(span_vecs).unsqueeze(-2) + self.right(span_vecs).unsqueeze(-3) + self.prod(p) + self.dist(d)
        h = self.dropout(torch.relu(h))

        out_res = self.scorer(h)
        return out_res


class OptFFpairsCorefLinkerNaive(nn.Module):
    """Just super-naive approach where a single nnet is used for both mention-mention and mention-entity combination.
    In theory, should perform worse than OptFFpairsLinkerCorefBase.
    """

    def __init__(self, dim_input, dim_input_entities, dim_output, config, span_pair_generator,
                 filter_singletons_with_matrix=False):
        super(OptFFpairsCorefLinkerNaive, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.filter_singletons_with_matrix = filter_singletons_with_matrix
        self.span_pair_generator = span_pair_generator
        self.left_spans = nn.Linear(dim_input, hidden_dim)
        self.right_spans = nn.Linear(dim_input, hidden_dim)
        self.prod_spans = nn.Linear(dim_input, hidden_dim)
        self.dist_spans = nn.Linear(span_pair_generator.dim_distance_embedding, hidden_dim)

        self.left_entities = nn.Linear(dim_input_entities, hidden_dim)

        self.dp1 = nn.Dropout(hidden_dp)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dp2 = nn.Dropout(hidden_dp)
        self.out = nn.Linear(hidden_dim, dim_output)

    def forward(self, span_vecs, entity_vecs, span_begin, span_end):
        # span_vecs: torch.Size([1, 9, 9, 150])
        # entity_vecs: torch.Size([1, 9, 17, 200])
        p = self.span_pair_generator.get_product_embedding(span_vecs)  # torch.Size([1, 9, 9, 1676])
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)  # torch.Size([1, 9, 9, 20])

        h_inter_span = self.left_spans(span_vecs).unsqueeze(-2) + self.right_spans(span_vecs).unsqueeze(-3) + \
                       self.prod_spans(p) + self.dist_spans(d)  # torch.Size([1, 9, 9, 150])

        h_span_ent = (self.left_entities(entity_vecs) + self.right_spans(span_vecs).unsqueeze(-2))
        # torch.Size([1, 9, 17, 150])

        h_final = torch.cat([h_span_ent, h_inter_span], dim=-2)  # torch.Size([1, 9, 26, 150])

        h_final = self.dp1(torch.relu(h_final))
        h_final = self.layer2(h_final)
        h_final = self.dp2(torch.relu(h_final))
        return self.out(h_final)


class OptFFpairsCorefLinkerBase(nn.Module):
    """The baseline implementation that, according to Johannes (18-09-2020), always works good which consists in:
        - concatenation of mention and entity embedding.
        - using separate nnet on mention-entity embedding from the nnet used on mention-mention embeddings.
    """

    def __init__(self, dim_input, entity_embedder, dim_output, config, span_pair_generator,
                 filter_singletons_with_matrix=False, dictionaries=None):
        super(OptFFpairsCorefLinkerBase, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        dim_input_entities = entity_embedder.dim
        self.dictionaries = dictionaries
        self.entity_embedder = entity_embedder

        self.span_pair_generator = span_pair_generator
        self.left_spans = nn.Linear(dim_input, hidden_dim)
        self.right_spans = nn.Linear(dim_input, hidden_dim)
        self.prod_spans = nn.Linear(dim_input, hidden_dim)
        self.dist_spans = nn.Linear(span_pair_generator.dim_distance_embedding, hidden_dim)

        self.left_entities = nn.Linear(dim_input_entities, hidden_dim)

        self.dp1_coref = nn.Dropout(hidden_dp)
        self.layer2_coref = nn.Linear(hidden_dim, hidden_dim)
        self.dp2_coref = nn.Dropout(hidden_dp)
        self.out_coref = nn.Linear(hidden_dim, dim_output)

        self.dp1_linking = nn.Dropout(hidden_dp)
        self.layer2_linking = nn.Linear(hidden_dim, hidden_dim)
        self.dp2_linking = nn.Dropout(hidden_dp)
        self.out_linking = nn.Linear(hidden_dim, dim_output)

        self.filter_singletons_with_matrix = filter_singletons_with_matrix
        self.filter_with_matrix_type = config['filter_with_matrix_type']
        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            # linear transformation from dim_input to hidden_dim, similar to left_spans, right_spans, prod_spans, etc..
            self.all_spans = nn.Linear(dim_input, hidden_dim)
            self.dp1_is_not_mention = nn.Dropout(hidden_dp)
            self.layer2_is_not_mention = nn.Linear(hidden_dim, hidden_dim)
            self.dp2_is_not_mention = nn.Dropout(hidden_dp)
            self.out_is_not_mention = nn.Linear(hidden_dim, dim_output)

    def forward(self, span_vecs, entity_vecs, span_begin, span_end):
        # span_vecs: torch.Size([1, 9, 9, 150])
        # entity_vecs: torch.Size([1, 9, 17, 200])
        p = self.span_pair_generator.get_product_embedding(span_vecs)  # torch.Size([1, 21, 21, 2324])
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)  # torch.Size([1, 21, 21, 20])

        h_inter_span = self.left_spans(span_vecs).unsqueeze(-2) + self.right_spans(span_vecs).unsqueeze(-3) + \
                       self.prod_spans(p) + self.dist_spans(d)  # torch.Size([1, 21, 21, 150])

        h_span_ent = (self.left_entities(entity_vecs) + self.right_spans(span_vecs).unsqueeze(-2))
        # torch.Size([1, 21, 21, 150])

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'entity_nnet':
            no_mention_col = torch.zeros((entity_vecs.shape[0], entity_vecs.shape[1]), dtype=torch.long,
                                         requires_grad=False, device=settings.device)
            no_mention_col[:, :] = self.dictionaries['entities'].lookup('NONE')
            no_mention_col_emb = self.entity_embedder(no_mention_col).unsqueeze(-2)
            h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans(span_vecs).unsqueeze(-2))
            h_span_ent = torch.cat([h_span_ent_nm, h_span_ent], dim=-2)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'coref_nnet':
            no_mention_col = torch.zeros((entity_vecs.shape[0], entity_vecs.shape[1]), dtype=torch.long,
                                         requires_grad=False, device=settings.device)
            no_mention_col[:, :] = self.dictionaries['entities'].lookup('NONE')
            no_mention_col_emb = self.entity_embedder(no_mention_col).unsqueeze(-2)

            h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans(span_vecs).unsqueeze(-2))
            h_inter_span = torch.cat([h_span_ent_nm, h_inter_span], dim=-2)

        # previous version (span-based coreference)
        h_inter_span = self.dp1_coref(torch.relu(h_inter_span))
        h_inter_span = self.layer2_coref(h_inter_span)
        h_inter_span = self.dp2_coref(torch.relu(h_inter_span))
        out_coref_scores = self.out_coref(h_inter_span)  # .shape --> torch.Size([1, 21, 21, 1])

        h_span_ent = self.dp1_linking(torch.relu(h_span_ent))
        h_span_ent = self.layer2_linking(h_span_ent)
        h_span_ent = self.dp2_linking(torch.relu(h_span_ent))
        out_linking_scores = self.out_linking(h_span_ent)  # .shape --> torch.Size([1, 21, 16, 1])

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            h_all_spans = self.all_spans(span_vecs.unsqueeze(-2))
            h_all_spans = self.dp1_is_not_mention(torch.relu(h_all_spans))
            h_all_spans = self.layer2_is_not_mention(h_all_spans)
            h_all_spans = self.dp2_is_not_mention(h_all_spans)
            out_is_not_mention_scores = self.out_is_not_mention(h_all_spans)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'coref_nnet':
            out_final_scores = torch.cat(
                [out_coref_scores[:, :, :1, :], out_linking_scores, out_coref_scores[:, :, 1:, :]], dim=-2)
        else:
            out_final_scores = torch.cat([out_linking_scores, out_coref_scores], dim=-2)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            out_final_scores = torch.cat([out_is_not_mention_scores, out_final_scores], dim=-2)

        return out_final_scores  # .shape --> torch.Size([1, 21, 37, 1])


class OptFFpairsCorefLinkerBaseHoi(nn.Module):
    """This version is similar to OptFFpairsCorefLinkerBase, but allows also for configurable
    nnet depth as well as nonlinearities in intermediate layers (i.e., Relu).
    """

    # def make_linear(self, in_features, out_features, bias=True, std=0.02):
    def make_linear(self, in_features, out_features, bias=True):
        linear = nn.Linear(in_features, out_features, bias)
        if self.init_weights is not None:
            init.normal_(linear.weight, std=self.init_weights)

        if bias and self.init_zeros_bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size, use_nonlinearity=False):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]

        if use_nonlinearity:
            ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        else:
            ffnn = [self.make_linear(feat_size, hidden_size[0]), self.dropout]

        for i in range(1, len(hidden_size)):
            if use_nonlinearity:
                ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
            else:
                ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), self.dropout]

        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, entity_embedder, dim_output, config, span_pair_generator,
                 filter_singletons_with_matrix=False, dictionaries=None):
        # TODO: we are here dim_input_entities still has to be passed
        super(OptFFpairsCorefLinkerBaseHoi, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.dropout = nn.Dropout(hidden_dp)
        # the components are self.left_spans, self.right_spans, self.prod_spans, and self.dist_spans, normally 1 layer
        self.components_ffnn_depth = config['components_ffnn_depth']
        self.use_nonlinearity_components = config['use_nonlinearity_components']

        self.scorers_ffnn_depth = config['scorers_ffnn_depth']
        self.use_nonlinearity_scorers = config['use_nonlinearity_scorers']

        self.init_weights = config['init_weights_std']
        self.init_zeros_bias = config['init_zeros_bias']

        dim_input_entities = entity_embedder.dim
        self.dictionaries = dictionaries
        self.entity_embedder = entity_embedder

        self.span_pair_generator = span_pair_generator

        self.separate_right_spans_for_ent = config['separate_right_spans_for_ent']

        self.left_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth,
                                         output_size=hidden_dim,
                                         use_nonlinearity=self.use_nonlinearity_components)

        self.right_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth,
                                          output_size=hidden_dim,
                                          use_nonlinearity=self.use_nonlinearity_components)

        self.prod_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth,
                                         output_size=hidden_dim,
                                         use_nonlinearity=self.use_nonlinearity_components)

        self.dist_spans = self.make_ffnn(span_pair_generator.dim_distance_embedding,
                                         [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim,
                                         use_nonlinearity=self.use_nonlinearity_components)

        self.left_entities = self.make_ffnn(dim_input_entities,
                                            [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim,
                                            use_nonlinearity=self.use_nonlinearity_components)

        if self.separate_right_spans_for_ent:
            self.right_spans_ent = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth,
                                                  output_size=hidden_dim,
                                                  use_nonlinearity=self.use_nonlinearity_components)

        self.scorer_coref = self.make_ffnn(hidden_dim, [hidden_dim] * self.scorers_ffnn_depth,
                                           output_size=dim_output,
                                           use_nonlinearity=self.use_nonlinearity_scorers)

        self.scorer_linker = self.make_ffnn(hidden_dim, [hidden_dim] * self.scorers_ffnn_depth,
                                            output_size=dim_output,
                                            use_nonlinearity=self.use_nonlinearity_scorers)

        self.filter_singletons_with_matrix = filter_singletons_with_matrix
        self.filter_with_matrix_type = config['filter_with_matrix_type']
        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            # linear transformation from dim_input to hidden_dim, similar to left_spans, right_spans, prod_spans, etc..
            self.all_spans = nn.Linear(dim_input, hidden_dim)
            self.dp1_is_not_mention = nn.Dropout(hidden_dp)
            self.layer2_is_not_mention = nn.Linear(hidden_dim, hidden_dim)
            self.dp2_is_not_mention = nn.Dropout(hidden_dp)
            self.out_is_not_mention = nn.Linear(hidden_dim, dim_output)

    def forward(self, span_vecs, entity_vecs, span_begin, span_end, do_only_coref):
        # span_vecs: torch.Size([1, 9, 9, 150])
        # entity_vecs: torch.Size([1, 9, 17, 200])
        prod_emb = self.span_pair_generator.get_product_embedding(span_vecs)  # torch.Size([1, 9, 9, 1676])
        dist_emb = self.span_pair_generator.get_distance_embedding(span_begin, span_end)  # torch.Size([1, 9, 9, 20])

        h_inter_span = self.left_spans(span_vecs).unsqueeze(-2) + self.right_spans(span_vecs).unsqueeze(-3) + \
                       self.prod_spans(prod_emb) + self.dist_spans(dist_emb)  # torch.Size([1, 9, 9, 150])

        if self.separate_right_spans_for_ent:
            h_span_ent = (self.left_entities(entity_vecs) + self.right_spans_ent(span_vecs).unsqueeze(-2))
        else:
            h_span_ent = (self.left_entities(entity_vecs) + self.right_spans(span_vecs).unsqueeze(-2))
        # torch.Size([1, 9, 17, 150])

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'entity_nnet':
            no_mention_col = torch.zeros((entity_vecs.shape[0], entity_vecs.shape[1]), dtype=torch.long,
                                         requires_grad=False, device=settings.device)
            no_mention_col[:, :] = self.dictionaries['entities'].lookup('NONE')
            no_mention_col_emb = self.entity_embedder(no_mention_col).unsqueeze(-2)
            if self.separate_right_spans_for_ent:
                h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans_ent(span_vecs).unsqueeze(-2))
            else:
                h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans(span_vecs).unsqueeze(-2))
            h_span_ent = torch.cat([h_span_ent_nm, h_span_ent], dim=-2)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'coref_nnet':
            no_mention_col = torch.zeros((entity_vecs.shape[0], entity_vecs.shape[1]), dtype=torch.long,
                                         requires_grad=False, device=settings.device)
            no_mention_col[:, :] = self.dictionaries['entities'].lookup('NONE')
            no_mention_col_emb = self.entity_embedder(no_mention_col).unsqueeze(-2)
            if self.separate_right_spans_for_ent:
                h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans_ent(span_vecs).unsqueeze(-2))
            else:
                h_span_ent_nm = (self.left_entities(no_mention_col_emb) + self.right_spans(span_vecs).unsqueeze(-2))

            h_inter_span = torch.cat([h_span_ent_nm, h_inter_span], dim=-2)

        h_inter_span = self.dropout(torch.relu(h_inter_span))
        out_coref_scores = self.scorer_coref(h_inter_span)  # --> .shape --> [1, 21, 21, 1]
        if do_only_coref:
            return out_coref_scores

        h_span_ent = self.dropout(torch.relu(h_span_ent))
        out_linking_scores = self.scorer_linker(h_span_ent)  # --> .shape --> [1, 21, 16, 1]

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            # h_all_spans = self.all_spans(span_vecs)
            h_all_spans = self.all_spans(span_vecs.unsqueeze(-2))
            h_all_spans = self.dp1_is_not_mention(torch.relu(h_all_spans))
            h_all_spans = self.layer2_is_not_mention(h_all_spans)
            h_all_spans = self.dp2_is_not_mention(h_all_spans)
            out_is_not_mention_scores = self.out_is_not_mention(h_all_spans)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'coref_nnet':
            out_final_scores = torch.cat(
                [out_coref_scores[:, :, :1, :], out_linking_scores, out_coref_scores[:, :, 1:, :]], dim=-2)
        else:
            out_final_scores = torch.cat([out_linking_scores, out_coref_scores], dim=-2)

        if self.filter_singletons_with_matrix and self.filter_with_matrix_type == 'separate_nnet':
            out_final_scores = torch.cat([out_is_not_mention_scores, out_final_scores], dim=-2)

        return out_final_scores  # .shape --> [1, 21, 37, 1]


class OptFFpairsCorefLinkerMTTBaseHoi(nn.Module):
    """The baseline implementation to produce the scores for the matrix to be used with Matrix Tree Theorem.
    """

    def make_linear(self, in_features, out_features, bias=True):
        linear = nn.Linear(in_features, out_features, bias)
        if self.init_weights_std is not None:
            init.normal_(linear.weight, std=self.init_weights_std)

        if bias and self.init_zeros_bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]

        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]

        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]

        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, entity_embedder, dim_output, config, span_pair_generator, dictionaries=None,
                 zeros_to_clusters=False, zeros_to_links=False):
        super(OptFFpairsCorefLinkerMTTBaseHoi, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4
        self.zeros_to_clusters = zeros_to_clusters
        self.zeros_to_links = zeros_to_links

        dim_input_entities = entity_embedder.dim

        self.init_weights_std = config['init_weights_std']
        self.init_root_type = config['init_root_type']
        self.init_root_std = config['init_root_std']
        self.root_requires_grad = config['root_requires_grad']
        self.apply_root_dropout = config['apply_root_dropout']

        self.dropout = nn.Dropout(hidden_dp)

        if not self.zeros_to_clusters:
            if self.init_root_type == 'std':
                self.root_embedding = torch.zeros(dim_input_entities, device=settings.device,
                                                  requires_grad=config['root_requires_grad'])
                if self.init_root_std is not None:
                    init.normal_(self.root_embedding, std=self.init_root_std)
                else:
                    ent_emb_mean = entity_embedder.embed.weight.mean().item()
                    ent_emb_std = entity_embedder.embed.weight.std().item()
                    init.normal_(self.root_embedding, mean=ent_emb_mean, std=ent_emb_std)
            elif self.init_root_type == 'ones':
                self.root_embedding = torch.ones(dim_input_entities, device=settings.device,
                                                 requires_grad=config['root_requires_grad'])
            elif self.init_root_type == 'zeros':
                self.root_embedding = torch.zeros(dim_input_entities, device=settings.device,
                                                  requires_grad=config['root_requires_grad'])
            else:
                raise RuntimeError('Not recognized init_root_type in OptFFpairsCorefLinkerMTTBaseHoi: ' +
                                   self.init_root_type)

        self.dictionaries = dictionaries
        self.entity_embedder = entity_embedder

        self.span_pair_generator = span_pair_generator
        self.components_ffnn_depth = config['components_ffnn_depth']
        self.scorers_ffnn_depth = config['scorers_ffnn_depth']
        self.init_zeros_bias = config['init_zeros_bias']
        self.separate_right_spans_for_ent = config['separate_right_spans_for_ent']

        self.left_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim)

        self.right_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim)

        if self.separate_right_spans_for_ent:
            self.right_spans_ent = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth,
                                                  output_size=hidden_dim)

        self.prod_spans = self.make_ffnn(dim_input, [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim)
        self.dist_spans = self.make_ffnn(span_pair_generator.dim_distance_embedding,
                                         [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim)
        self.left_entities = self.make_ffnn(dim_input_entities,
                                            [hidden_dim] * self.components_ffnn_depth, output_size=hidden_dim)

        self.scorer_coref = self.make_ffnn(hidden_dim, [hidden_dim] * self.scorers_ffnn_depth, output_size=dim_output)

        self.scorer_linker = self.make_ffnn(hidden_dim, [hidden_dim] * self.scorers_ffnn_depth, output_size=dim_output)

    def forward(self, span_vecs, entity_vecs, span_begin, span_end, candidate_lengths, max_cand_length):
        # span_vecs: torch.Size([1, 9, 9, 150])
        # entity_vecs: torch.Size([1, 9, 17, 200])
        p = self.span_pair_generator.get_product_embedding(span_vecs)  # torch.Size([1, 9, 9, 1676])
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)  # torch.Size([1, 9, 9, 20])

        h_inter_span = self.left_spans(span_vecs).unsqueeze(-2) + self.right_spans(span_vecs).unsqueeze(-3) + \
                       self.prod_spans(p) + self.dist_spans(d)  # torch.Size([1, 9, 9, 150])

        # TODO: ideas to make this expansion cleaner?
        # todo: MAYBE USE EXPAND??: https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969/2
        if not self.zeros_to_clusters:
            rooted_entity_vecs = self.root_embedding.repeat(entity_vecs.shape[0], entity_vecs.shape[1], 1, 1)
            if self.root_requires_grad and self.apply_root_dropout:  # dropout is applied if the root embeddings are learned
                rooted_entity_vecs = self.dropout(rooted_entity_vecs)

            if max_cand_length == 0:
                # if there are no candidates, no need to concatenate, with root is enough
                entity_vecs = rooted_entity_vecs
            else:
                entity_vecs = torch.cat([rooted_entity_vecs, entity_vecs], dim=-2)
                # entity_vecs.shape --> torch.Size([1, 21, 17, 200])
            if self.separate_right_spans_for_ent:
                h_span_ent = (self.left_entities(entity_vecs) + self.right_spans_ent(span_vecs).unsqueeze(-2))
            else:
                h_span_ent = (self.left_entities(entity_vecs) + self.right_spans(span_vecs).unsqueeze(-2))
        else:
            if max_cand_length == 0:
                h_span_ent = None
            else:
                if self.separate_right_spans_for_ent:
                    h_span_ent = (self.left_entities(entity_vecs) + self.right_spans_ent(span_vecs).unsqueeze(-2))
                else:
                    h_span_ent = (self.left_entities(entity_vecs) + self.right_spans(span_vecs).unsqueeze(-2))

        h_inter_span = self.dropout(torch.relu(h_inter_span))
        out_coref_scores = self.scorer_coref(h_inter_span)

        h_span_ent = self.dropout(torch.relu(h_span_ent))
        out_linking_scores = self.scorer_linker(h_span_ent)

        out_final_scores = torch.cat([out_linking_scores, out_coref_scores], dim=-2)
        # out_final_scores.shape --> torch.Size([1, 21, 38, 1])

        if self.zeros_to_clusters:
            # adds the zeros to cluster
            zeros_col = torch.zeros((out_final_scores.shape[0], out_final_scores.shape[1], 1, 1),
                                    device=settings.device)
            out_final_scores = torch.cat([zeros_col, out_final_scores], dim=-2)
        return out_final_scores


class DotPairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(DotPairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.left = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.right = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, span_vecs, span_begin, span_end):
        l = self.left(span_vecs)  # [batch, length, dim_hidden]
        r = self.right(span_vecs)  # [batch, length, dim_hidden]
        s = torch.matmul(l, r.permute(0, 2, 1))
        return s.unsqueeze(-1)
