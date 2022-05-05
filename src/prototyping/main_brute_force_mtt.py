import itertools
import json
import math
import random
from typing import Set, List

import numpy as np
import torch


def get_all_spanning_trees_v2(nr_nodes: int = 6, mask_matrix=None):
    # this version  puts "root" node in the matrix

    node_letter_names = 'RABCDEFGHIJKLMNOPQRSTUVWXUZ1234567890~!@#$%^&*()_-+=}]{[:;"\'\\,<.>/?'

    idx_to_node = []
    node_to_idx = dict()
    # idx_to_node = ['R']
    # node_to_idx = {'R': 0}

    # for id in range(1, nr_spans + 1):
    for id in range(0, nr_nodes):
        idx_to_node.append(node_letter_names[id])
        node_to_idx[node_letter_names[id]] = id
        # idx_to_node.append(node_letter_names[id - 1])
        # node_to_idx[node_letter_names[id - 1]] = id

    # tensors_mention_idxs = [torch.tensor(list(range(nr_nodes + 1))) for _ in range(nr_nodes)]
    tensors_mention_idxs = [torch.tensor(list(range(nr_nodes))) for _ in range(nr_nodes - 1)]
    cart_prod_idx = torch.cartesian_prod(*tensors_mention_idxs)
    if len(cart_prod_idx.shape) == 1:
        cart_prod_idx = cart_prod_idx.unsqueeze(-1)
    # tensor([0, 1]) --> .shape --> torch.Size([2])
    # tensor([[0, 0],
    #         [0, 1],
    #         [0, 2],
    #         [1, 0],
    #         [1, 1],
    #         [1, 2],
    #         [2, 0],
    #         [2, 1],
    #         [2, 2]])
    # -->shape --> torch.Size([9, 2])
    adjacency = torch.zeros(cart_prod_idx.size(0), cart_prod_idx.size(1) + 1, cart_prod_idx.size(1),
                            dtype=torch.int)
    # adjacency = torch.zeros(cart_prod_idx.size(0), cart_prod_idx.size(1), cart_prod_idx.size(1) + 1,
    #                         dtype=torch.int)

    cart_prod_idx2 = cart_prod_idx.reshape(-1) + torch.arange(
        cart_prod_idx.size(0) * cart_prod_idx.size(1)) * adjacency.size(1)
    # cart_prod_idx.size(0) * cart_prod_idx.size(1)) * adjacency.size(2)
    adj_temp = adjacency.permute(0, 2, 1).contiguous()
    # adjacency.view(-1)[cart_prod_idx2] = 1
    adj_temp.view(-1)[cart_prod_idx2] = 1

    adjacency = adj_temp.permute(0, 2, 1).contiguous()

    # concatenates the row to root with all zeros, since the root doesn't get edges pointing to it
    empty_col_to_concat = torch.zeros(adjacency.size(0), adjacency.size(1), 1,
                                      dtype=torch.int)

    adjacency = torch.cat([empty_col_to_concat, adjacency], dim=2)

    # now counts the unique spanning trees
    nr_spanning_trees = 0
    print('adjacency shape: ', adjacency.shape)
    nr_processed = 0
    spanning_trees = []
    for curr_adj_matrix in adjacency:
        nr_processed += 1
        if nr_processed % 1000 == 0:
            print('nr pre-processed: ', nr_processed, '  nr_kept (spanning trees without loops): ', nr_spanning_trees)
        # continue if the root doesn't have outbound relations
        # if curr_adj_matrix[:, 0].sum() == 0:
        # curr_adj_matrix = curr_adj_matrix.T
        # curr_adj_matrix = curr_adj_matrix[1:, :][:, 1:]  # no root element in the matrix for now
        if mask_matrix is not None:
            curr_adj_matrix = curr_adj_matrix * mask_matrix
            # after applying mask there should be a minimum number of edges
            if curr_adj_matrix.sum().item() < nr_nodes - 1:
                # print('missing necessary edges, continuing: ')
                # print(curr_adj_matrix)
                continue

        if curr_adj_matrix[0, :].sum() == 0:
            continue
        # continue if there are relations to itself (main diagonal)
        if torch.diagonal(curr_adj_matrix, 0).sum() > 0:
            continue

        # continue if symmetries detected (ex: A -> B ; B -> A
        upper_triangle = torch.triu(curr_adj_matrix)
        lower_triangle = torch.tril(curr_adj_matrix)
        lower_triangle_t = lower_triangle.transpose(1, 0)
        symm_check = upper_triangle + lower_triangle_t
        if torch.max(symm_check) > 1:
            # print('continuing because of symmetries detected')
            # print(curr_adj_matrix)
            continue

        # continue if a larger cycle is found such as in (R, A, B, C, D): R->A ; B->C; C->D; D->B
        # use both trace and brute force to detect this, details on:
        # https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix#:~:text=Start%20from%20an%20edge%20(i,then%20a%20cycle%20is%20detected
        # check if from the root all the nodes can be accessed using tree traversal
        # traversed_root = (curr_adj_matrix[:, 0] == 1).nonzero()
        # for curr_entries in range(nr_spans):
        traversed_nodes = list()
        # adds the root to traversed nodes
        traversed_nodes.append(idx_to_node[0])
        to_traverse_nodes = list()
        for idx, curr_entry in enumerate(curr_adj_matrix[0, :].tolist()):
            if curr_entry == 1:
                traversed_nodes.append(idx_to_node[idx])
                to_traverse_nodes.append(idx_to_node[idx])
        if len(traversed_nodes) == 0:
            continue

        # for idx_root in traversed_root:
        #     traversed_nodes.add(idx_to_node[idx_root[0]])

        def recursive_tree_traverse(traversed_nodes: Set, to_traverse_nodes: List):
            curr_to_traverse_to = set()
            for to_traverse in to_traverse_nodes:
                idx = node_to_idx[to_traverse]
                # pointed_to = (curr_adj_matrix[:, idx] == 1).nonzero()
                pointed_to = torch.nonzero(curr_adj_matrix[idx, :] == 1, as_tuple=False)
                for curr_pointed_node_idx in pointed_to:
                    curr_pointed_node = idx_to_node[curr_pointed_node_idx[0]]
                    if curr_pointed_node in traversed_nodes:
                        return False  # loop detected!
                    else:
                        traversed_nodes.add(curr_pointed_node)
                        curr_to_traverse_to.add(curr_pointed_node)
            if len(curr_to_traverse_to) > 0:
                return recursive_tree_traverse(traversed_nodes, list(curr_to_traverse_to))
            return True

        traversed_nodes_s = set(traversed_nodes)
        recursive_tree_traverse(traversed_nodes_s, to_traverse_nodes=to_traverse_nodes)

        if len(traversed_nodes_s) < len(idx_to_node):
            # print('loop detected, continuing')
            continue
        else:
            spanning_trees.append({'matrix': curr_adj_matrix})
            nr_spanning_trees += 1
        # spanning_trees.append({'entry': curr_entries, 'matrix': curr_adj_matrix})
        # nr_spanning_trees += 1

    print('total number of distinct spanning trees calculated: ', nr_spanning_trees)
    return spanning_trees


def get_all_spanning_trees(nr_spans: int = 5, mask_matrix=None):
    # a = [0, 1, 2, 3, 4, 5]
    # b = [0, 1, 2, 3, 4, 5]
    # c = [0, 1, 2, 3, 4, 5]
    # d = [0, 1, 2, 3, 4, 5]
    # e = [0, 1, 2, 3, 4, 5]
    #
    # idx_to_node = ['R', 'A', 'B', 'C', 'D', 'E']
    # node_to_idx = {'R': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    # tensor_a = torch.tensor(a)
    # tensor_b = torch.tensor(b)
    # tensor_c = torch.tensor(c)
    # tensor_d = torch.tensor(d)
    # tensor_e = torch.tensor(e)

    # cart_prod_idx = torch.cartesian_prod(tensor_a, tensor_b, tensor_c, tensor_d, tensor_e)

    node_letter_names = 'ABCDEFGHIJKLMNOPQRSTUVWXUZ'

    idx_to_node = []
    node_to_idx = dict()
    # idx_to_node = ['R']
    # node_to_idx = {'R': 0}

    # for id in range(1, nr_spans + 1):
    for id in range(0, nr_spans):
        idx_to_node.append(node_letter_names[id])
        node_to_idx[node_letter_names[id]] = id
        # idx_to_node.append(node_letter_names[id - 1])
        # node_to_idx[node_letter_names[id - 1]] = id

    tensors_mention_idxs = [torch.tensor(list(range(nr_spans + 1))) for _ in range(nr_spans)]
    cart_prod_idx = torch.cartesian_prod(*tensors_mention_idxs)

    adjacency = torch.zeros(cart_prod_idx.size(0), cart_prod_idx.size(1), cart_prod_idx.size(1) + 1,
                            dtype=torch.int)

    cart_prod_idx2 = cart_prod_idx.reshape(-1) + torch.arange(
        cart_prod_idx.size(0) * cart_prod_idx.size(1)) * adjacency.size(2)

    adjacency.view(-1)[cart_prod_idx2] = 1

    # concatenates the row to root with all zeros, since the root doesn't get edges pointing to it
    empty_row_to_concat = torch.zeros(adjacency.size(0), 1, adjacency.size(2),
                                      dtype=torch.int)
    adjacency = torch.cat([empty_row_to_concat, adjacency], dim=1)
    # now counts the unique spanning trees
    nr_spanning_trees = 0
    print('adjacency shape: ', adjacency.shape)
    nr_processed = 0
    spanning_trees = []
    for curr_adj_matrix in adjacency:
        nr_processed += 1
        if nr_processed % 1000 == 0:
            print('nr pre-processed: ', nr_processed, '  nr_kept (spanning trees without loops): ', nr_spanning_trees)
        # continue if the root doesn't have outbound relations
        # if curr_adj_matrix[:, 0].sum() == 0:
        curr_adj_matrix = curr_adj_matrix.T
        curr_adj_matrix = curr_adj_matrix[1:, :][:, 1:]  # no root element in the matrix for now
        if mask_matrix is not None:
            curr_adj_matrix = curr_adj_matrix * mask_matrix
            # after applying mask there should be a minimum number of edges
            if curr_adj_matrix.sum().item() < nr_spans - 1:
                # print('missing necessary edges, continuing: ')
                # print(curr_adj_matrix)
                continue

        if curr_adj_matrix[0, :].sum() == 0:
            continue
        # continue if there are relations to itself (main diagonal)
        if torch.diagonal(curr_adj_matrix, 0).sum() > 0:
            continue

        # continue if symmetries detected (ex: A -> B ; B -> A
        upper_triangle = torch.triu(curr_adj_matrix)
        lower_triangle = torch.tril(curr_adj_matrix)
        lower_triangle_t = lower_triangle.transpose(1, 0)
        symm_check = upper_triangle + lower_triangle_t
        if torch.max(symm_check) > 1:
            # print('continuing because of symmetries detected')
            # print(curr_adj_matrix)
            continue

        # continue if a larger cycle is found such as in (R, A, B, C, D): R->A ; B->C; C->D; D->B
        # use both trace and brute force to detect this, details on:
        # https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix#:~:text=Start%20from%20an%20edge%20(i,then%20a%20cycle%20is%20detected
        # check if from the root all the nodes can be accessed using tree traversal
        # traversed_root = (curr_adj_matrix[:, 0] == 1).nonzero()
        # for curr_entries in range(nr_spans):
        entries_combinations = list(itertools.product([0, 1], repeat=nr_spans))
        for curr_entries in entries_combinations:
            # traversed_root = (curr_adj_matrix[0, :] == 1).nonzero()
            # traversed_root = (curr_adj_matrix[curr_entries, :] == 1).nonzero()
            traversed_nodes = list()
            for idx, curr_entry in enumerate(curr_entries):
                if curr_entry == 1:
                    traversed_nodes.append(idx_to_node[idx])
            if len(traversed_nodes) == 0:
                continue

            # for idx_root in traversed_root:
            #     traversed_nodes.add(idx_to_node[idx_root[0]])

            def recursive_tree_traverse(traversed_nodes: Set, to_traverse_nodes: List):
                curr_to_traverse_to = set()
                for to_traverse in to_traverse_nodes:
                    idx = node_to_idx[to_traverse]
                    # pointed_to = (curr_adj_matrix[:, idx] == 1).nonzero()
                    pointed_to = torch.nonzero(curr_adj_matrix[idx, :] == 1, as_tuple=False)
                    for curr_pointed_node_idx in pointed_to:
                        curr_pointed_node = idx_to_node[curr_pointed_node_idx[0]]
                        if curr_pointed_node in traversed_nodes:
                            return False  # loop detected!
                        else:
                            traversed_nodes.add(curr_pointed_node)
                            curr_to_traverse_to.add(curr_pointed_node)
                if len(curr_to_traverse_to) > 0:
                    return recursive_tree_traverse(traversed_nodes, list(curr_to_traverse_to))
                return True

            traversed_nodes_s = set(traversed_nodes)
            recursive_tree_traverse(traversed_nodes_s, to_traverse_nodes=traversed_nodes.copy())

            if len(traversed_nodes_s) < len(idx_to_node):
                continue
            else:
                spanning_trees.append({'entry': curr_entries, 'matrix': curr_adj_matrix})
                nr_spanning_trees += 1
            # spanning_trees.append({'entry': curr_entries, 'matrix': curr_adj_matrix})
            # nr_spanning_trees += 1

    print('total number of distinct spanning trees calculated: ', nr_spanning_trees)
    return spanning_trees


def get_score_trees(spanning_trees, scores):
    tot_weight = 0.0
    for curr_spanning_tree in spanning_trees:
        # print('current spanning tree: ', curr_spanning_tree)
        curr_tree_scores = scores[curr_spanning_tree['matrix'].bool()]
        curr_tree_weight = torch.prod(curr_tree_scores).item()
        tot_weight += curr_tree_weight
    return tot_weight


def get_score_trees_cluster_restrictive(spanning_trees, scores, clusters):
    tot_weight = 0.0
    for curr_spanning_tree in spanning_trees:
        ignore_curr_tree = False
        curr_entry = torch.tensor(curr_spanning_tree['entry'], dtype=torch.int)
        for curr_cluster in clusters:
            t_cluster = torch.tensor(curr_cluster, dtype=torch.long)
            if curr_entry[t_cluster].sum() != 1:
                ignore_curr_tree = True

        if not ignore_curr_tree:
            curr_tree_weight = torch.prod(scores[curr_spanning_tree['matrix'].bool()]).item()
            tot_weight += curr_tree_weight
    return tot_weight


def get_scores_brute_restrictive_v2(spanning_trees, scores, clusters, link_idxs):
    """
    In this version the root is not in "entry" but rather encoded in the matrix.
    :param spanning_trees:
    :param scores:
    :param clusters:
    :return:
    """
    tot_weight = 0.0
    for curr_spanning_tree in spanning_trees:
        ignore_curr_tree = False
        # curr_entry = torch.tensor(curr_spanning_tree['entry'], dtype=torch.int)
        curr_entry = curr_spanning_tree['matrix']
        for curr_cluster in clusters:
            t_cluster = torch.tensor(curr_cluster, dtype=torch.long)
            if curr_entry[0, t_cluster].sum() != 1:
                ignore_curr_tree = True

        if not ignore_curr_tree:
            curr_tree_weight = torch.prod(scores[curr_spanning_tree['matrix'].bool()]).item()
            # print('Tree weight brute force (log space): ', math.log(curr_tree_weight))
            tot_weight += curr_tree_weight
    return tot_weight


def get_nr_trees_cluster_restrictive(spanning_trees, clusters):
    nr_trees = 0
    for curr_spanning_tree in spanning_trees:
        ignore_curr_tree = False
        curr_entry = torch.tensor(curr_spanning_tree['entry'], dtype=torch.int)
        for curr_cluster in clusters:
            t_cluster = torch.tensor(curr_cluster, dtype=torch.long)
            if curr_entry[t_cluster].sum() != 1:
                ignore_curr_tree = True

        if not ignore_curr_tree:
            nr_trees += 1
    return nr_trees


def get_nr_trees_brute_restrictive_v2(spanning_trees, clusters):
    """
    In this version the root is not in "entry" but rather encoded in the matrix.
    :param spanning_trees:
    :param clusters:
    :return:
    """
    nr_trees = 0
    for curr_spanning_tree in spanning_trees:
        ignore_curr_tree = False
        # curr_entry = torch.tensor(curr_spanning_tree['entry'], dtype=torch.int)
        # curr_entry = curr_spanning_tree['matrix'][0, 1:]
        curr_entry = curr_spanning_tree['matrix']

        for curr_cluster in clusters:
            t_cluster = torch.tensor(curr_cluster, dtype=torch.long)
            if curr_entry[0, t_cluster].sum() != 1:
                ignore_curr_tree = True

        if not ignore_curr_tree:
            nr_trees += 1
    return nr_trees


def get_score_trees_mtt(mask_matrix, scores):
    mask_matrix = mask_matrix.double()
    scores = scores.double()
    filtered_scores = mask_matrix * scores
    laplacian_scores = torch.eye(filtered_scores.shape[-2], filtered_scores.shape[-1])
    laplacian_scores = laplacian_scores.double()
    laplacian_scores = laplacian_scores * filtered_scores.sum(dim=-2)  # main diagonal
    laplacian_scores += (filtered_scores * -1.0)

    # mtt_scores = torch.slogdet(laplacian_scores)
    # mtt_scores = torch.det(laplacian_scores[1:, 1:])
    mtt_scores = torch.logdet(laplacian_scores[1:, 1:])
    # tensor(3.0862e+14, dtype=torch.float64)
    # torch.det((laplacian_scores[1:,1:])) = tensor(-4.3142e+13)
    # torch.logdet(laplacian_scores[1:,1:]) = {Tensor} tensor(nan)
    # torch.slogdet(laplacian_scores[1:,1:]) = {slogdet} torch.return_types.slogdet(\nsign=tensor(-1.),\nlogabsdet=tensor(31.3955))
    return mtt_scores


def get_score_trees_mtt_partition(mask_matrix, scores):
    mask_matrix = mask_matrix.double()
    scores = scores.double()
    filtered_scores = mask_matrix * scores

    # change a
    filtered_scores = filtered_scores[1:, :][:, 1:]

    laplacian_scores = torch.eye(filtered_scores.shape[-2], filtered_scores.shape[-1])
    laplacian_scores = laplacian_scores.double()
    laplacian_scores = laplacian_scores * filtered_scores.sum(dim=-2)  # main diagonal
    laplacian_scores += (filtered_scores * -1.0)

    # change a
    # laplacian_minor = laplacian_scores[1:, :][:, 1:]
    laplacian_minor = laplacian_scores

    # the laplacian's first row gets replaced
    laplacian_minor[0, :] = scores[0][1:]

    mtt_scores = torch.logdet(laplacian_minor)

    return mtt_scores


if __name__ == "__main__":

    ##################################### loading and analyzing some tensors that are causing problems in production

    # curr_pred_scores = torch.load('debug_curr_pred_scores.pt', map_location=torch.device('cpu'))
    # curr_targets_mask_loop = torch.load('debug_curr_targets_mask_loop.pt', map_location=torch.device('cpu'))
    #
    # curr_pred_scores = curr_pred_scores[0:12, :][:, 0:12]
    # curr_targets_mask_loop = curr_targets_mask_loop[0:12, :][:, 0:12]
    #
    # debug_score_mtt = get_score_trees_mtt(curr_targets_mask_loop, curr_pred_scores)
    #####################################

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    load_from_file = True

    if load_from_file:
        # file_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/config/local_tests/' \
        #             'hoi_spanbert_refactor1/coreflinker_mtt/debugging_logs/ep_0089_DW_187058777_debugging_mtt.json'
        # file_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/config/local_tests/' \
        #             'hoi_spanbert_refactor1/coreflinker_mtt/debugging_logs/ep_0019_DW_187058777_debugging_mtt.json'
        # file_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/config/local_tests/' \
        #             'hoi_spanbert_refactor1/coreflinker_mtt/debugging_logs/ep_0004_DW_187058777_debugging_mtt.json'
        file_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/config/local_tests/' \
                    'hoi_spanbert_refactor1/coreflinker_mtt/debugging_logs/ep_0004_DW_187058777_debugging_mtt.json'

        loaded_json = json.load(open(file_path, 'r'))
        scores = loaded_json['scores']
        clusters_gold = loaded_json['clusters_gold']
        items = loaded_json['items']
        mention_idx = set()
        link_idx = set()
        links_to_idxs = dict()
        for idx_item, curr_item in enumerate(items):
            if '@' in curr_item:
                mention_idx.add(idx_item)
            elif 'ROOT' not in curr_item:
                link_idx.add(idx_item)
                links_to_idxs[curr_item] = idx_item

        span_to_candidates = loaded_json['span_to_candidates']
        possible_link_spans_combinations = set()
        for curr_span_idx, curr_span_links in span_to_candidates.items():
            for curr_link in curr_span_links:
                possible_link_spans_combinations.add((links_to_idxs[curr_link], int(curr_span_idx)))

        tensor_scores = torch.DoubleTensor(scores)
        scores = tensor_scores
        clusters = clusters_gold
        flat_mention_ids = [item for sublist in clusters for item in sublist]
        nr_elements = len(items) - 1

    print('nr of elements: ', scores.shape[-1] - 1)  # -1 because of root

    mask_target_matrix = torch.zeros_like(scores, dtype=torch.int)
    mask_z_matrix = torch.zeros_like(scores, dtype=torch.int)
    mask_z_matrix[0, 1:] = 1  # from root can reach anywhere

    for curr_cluster in clusters:
        link_in_cluster = False
        # link_idx = -1
        for el1_cluster in curr_cluster:
            if el1_cluster in link_idx:
                mask_target_matrix[0, el1_cluster] = 1  # 1 from root to link
                link_in_cluster = True
            for el2_cluster in curr_cluster:
                if el1_cluster != el2_cluster:
                    if el1_cluster in mention_idx and el2_cluster in mention_idx:
                        mask_target_matrix[el1_cluster, el2_cluster] = 1
                    elif el1_cluster in link_idx and el2_cluster in mention_idx:
                        mask_target_matrix[el1_cluster, el2_cluster] = 1
        if not link_in_cluster:
            idx_torch_cluster = torch.tensor(curr_cluster, dtype=torch.long)
            mask_target_matrix[0, :][idx_torch_cluster] = 1  # 1 from root to all mentions in cluster ??
    for curr_link_idx in link_idx:
        # all the links have to be connected to root by definition in target
        mask_target_matrix[0,curr_link_idx] = 1

    # flat_el_clusters = [el for el in cl for cl in clusters for el in cl]
    flat_el_clusters = [item for sublist in clusters for item in sublist]

    for el1_cluster in flat_el_clusters:
        for el2_cluster in flat_el_clusters:
            if el1_cluster != el2_cluster:
                if el1_cluster in mention_idx and el2_cluster in mention_idx:
                    mask_z_matrix[el1_cluster, el2_cluster] = 1
                elif el1_cluster in link_idx and el2_cluster in mention_idx:
                    if (el1_cluster, el2_cluster) in possible_link_spans_combinations:
                        mask_z_matrix[el1_cluster, el2_cluster] = 1

    for curr_link_span_comb in possible_link_spans_combinations:
        mask_z_matrix[curr_link_span_comb[0],curr_link_span_comb[1]] = 1

    print('mask_mtt_matrix: ')
    print(mask_z_matrix)

    print('mask_target_matrix: ')
    print(mask_target_matrix)

    masked_tgt_scores = scores * mask_target_matrix
    masked_z_scores = scores * mask_z_matrix

    if load_from_file:
        scores = torch.arcsinh(scores)
        scores = torch.log(mask_z_matrix.float()) + scores
        scores = torch.exp(scores)

    # print('masked target scores: ')
    # print(masked_tgt_scores)
    # print('masked z scores: ')
    # print(masked_z_scores)

    all_spanning_trees = get_all_spanning_trees_v2(nr_elements + 1, mask_z_matrix)
    target_spanning_trees = get_all_spanning_trees_v2(nr_elements + 1, mask_target_matrix)

    score_brute_all = get_score_trees(all_spanning_trees, scores)
    score_mtt_all = get_score_trees_mtt(mask_z_matrix, scores)

    print('score of ALL spanning trees brute force: ', score_brute_all, ' and in log space: ',
          math.log(score_brute_all))
    print('score of ALL spanning trees MTT: ', score_mtt_all.item())

    # print('score of TARGET spanning trees brute force: ', score_brute_target, ' and in log space: ',
    #       math.log(score_brute_target))
    # print('score of TARGET spanning trees MTT: ', sc_trees_mtt.item())

    nr_connection_br_force = get_nr_trees_brute_restrictive_v2(target_spanning_trees, clusters)
    w_connection_br_force = get_scores_brute_restrictive_v2(target_spanning_trees, scores, clusters,
                                                            link_idxs=link_idx)
    # print('nr of TARGET spanning trees w. single connection to cluster brute force: ', nr_connection_br_force)
    print('score of TARGET spanning trees w. single connection from root to cluster for each cluster (brute force): ',
          w_connection_br_force, ' and in log space: ', math.log(w_connection_br_force))
    # print('weight of TARGET spanning trees w. single connection to cluster brute force: ',
    #       math.log(w_connection_br_force))

    tot_nil_scores_single_conn = 0.0
    nr_iters = 0

    nil_clusters = []
    not_nil_clusters = []
    for curr_cluster in clusters:
        curr_cluster_nil = True
        for curr_cl_element in curr_cluster:
            if curr_cl_element in link_idx:
                curr_cluster_nil = False
                break
        if curr_cluster_nil:
            nil_clusters.append(curr_cluster)
        else:
            not_nil_clusters.append(curr_cluster)

    # for curr_nil_entries in itertools.product(*clusters):
    nil_clusters_indices = [item for sublist in nil_clusters for item in sublist]
    nil_clusters_indices = [0] + nil_clusters_indices
    nil_clusters_indices = torch.tensor(nil_clusters_indices, dtype=torch.long)
    for curr_nil_entries in itertools.product(*nil_clusters):
        nr_iters += 1
        ind_root = torch.tensor(curr_nil_entries, dtype=torch.long)
        # ind_root = ind_root + 1  # accounting for the first root column
        mask_target_matrix_c = mask_target_matrix.clone().detach()
        mask_target_matrix_c[0, :] = 0
        mask_target_matrix_c[0, ind_root] = 1
        curr_score = get_score_trees_mtt(mask_target_matrix_c[nil_clusters_indices, :][:, nil_clusters_indices],
                                         scores[nil_clusters_indices, :][:, nil_clusters_indices])
        tot_nil_scores_single_conn += math.exp(curr_score)

    tot_not_nil_scores = torch.Tensor([0.0])
    for curr_not_nil_cluster in not_nil_clusters:
        ind_cluster = torch.tensor([0] + curr_not_nil_cluster, dtype=torch.long)
        curr_mask_target = mask_target_matrix[ind_cluster, :][:, ind_cluster]
        curr_scores = scores[ind_cluster, :][:, ind_cluster]
        curr_score = get_score_trees_mtt(curr_mask_target, curr_scores)
        # print('not nil cluster score log space: ', curr_score.item())
        if tot_not_nil_scores == 0.0:
            tot_not_nil_scores = curr_score
        else:
            tot_not_nil_scores = tot_not_nil_scores + curr_score

    tot_score = tot_not_nil_scores + math.log(tot_nil_scores_single_conn)
    # if not load_from_file:
    print('weight of TARGET NIL spanning trees w. single connection to cluster MTT: ', tot_nil_scores_single_conn,
          ' and in log space: ', math.log(tot_nil_scores_single_conn))

    print('weight of TARGET NOT NIL spanning trees w. single connection to cluster MTT (log space): ',
          tot_not_nil_scores.item())

    print('weight of TARGET spanning trees (including not nil) w. single connection to cluster MTT: ',
          torch.exp(tot_not_nil_scores) * tot_nil_scores_single_conn,
          ' and in log space: ', (tot_not_nil_scores + math.log(tot_nil_scores_single_conn)).item())

    tot_nil_scores_single_conn = 1
    # now trying to do it using mtt more efficiently for each cluster once (no need of cartesian product)
    for curr_nil_cluster in nil_clusters:
        ind_root = torch.tensor(curr_nil_cluster, dtype=torch.long)
        # ind_root = ind_root + 1  # accounting for the first root column
        # appends also the root
        ind_root = torch.cat([torch.tensor([0], dtype=torch.long), ind_root])
        scores_cluster = scores[ind_root, :][:, ind_root]
        mask_target_matrix_c = mask_target_matrix.clone().detach()
        mask_target_matrix_c[0, :] = 0
        mask_target_matrix_c[0, ind_root] = 1
        mask_target_matrix_c = mask_target_matrix_c[ind_root, :][:, ind_root]
        # root can not point to itself - "atadura con alambre"
        mask_target_matrix_c[0][0] = 0
        # curr_score = get_score_trees_mtt_partition(mask_target_matrix_c, scores)
        curr_score = get_score_trees_mtt_partition(mask_target_matrix_c, scores_cluster)
        # print('partition approach NIL cluster (log space): ', torch.log(curr_score).item())
        tot_nil_scores_single_conn *= curr_score

    print('NIL clusters experimental weight using linear partition approach: ', tot_nil_scores_single_conn,
          ' and in log space: ', math.log(tot_nil_scores_single_conn))
    ### implementing using product instead of cartesians combinations as explained by Johannes in notes of
    # https://docs.google.com/presentation/d/1Za2gCNq55gp1MCTlg4p0CN-JGyKjj4WtzxpYJKDZz3E/edit#slide=id.gd02f9cc825_0_419
    # tot_target_clusters_score = 0.0
    # for idx_cluster, cluster_mentions in enumerate(clusters):
    #     # print('inside the cluster ', idx_cluster, ': ', cluster_mentions)
    #     # gets the sub-matrix for weights to be calculated on
    #     nr_nodes = len(cluster_mentions) + 1
    #     cluster_target_mask = torch.zeros((nr_nodes, nr_nodes), dtype=torch.int)
    #
    #     mention_indices = torch.tensor(cluster_mentions, dtype=torch.long)
    #     # account for root node
    #     mention_indices = mention_indices + 1
    #     # TODO: alternatively, the target_matrix can be just 1s with 0s in the main diagonal
    #     # concatenates 0 to mention_indices for root
    #     mention_indices = torch.cat([torch.tensor([0], dtype=torch.int), mention_indices])
    #     target_matrix = mask_target_matrix[mention_indices, :][:, mention_indices]
    #     cluster_target_mask[:, :] = target_matrix
    #     cluster_scores = scores[mention_indices, :][:, mention_indices]
    #     curr_target_cluster_score = 0.0
    #     for idx_mention, mention_id in enumerate(cluster_mentions):
    #         cluster_target_mask[0, :] = 0
    #         # +1 because we account for root node
    #         cluster_target_mask[0, idx_mention + 1] = 1
    #
    #         score_mtt = get_score_trees_mtt(cluster_target_mask, cluster_scores)
    #         curr_target_cluster_score += score_mtt
    #     if tot_target_clusters_score > 0.0:
    #         tot_target_clusters_score = tot_target_clusters_score * curr_target_cluster_score
    #     else:
    #         tot_target_clusters_score = curr_target_cluster_score
    # cluster_target_mask = mask_target_matrix[]
    #
    # print('weight of TARGET spanning trees w. single connection to cluster MTT (product): ', tot_target_clusters_score)
    #
    # # nr of all possible spanning trees in the mask
    # print('nr of ALL possible spanning trees brute force: ', len(all_spanning_trees))
    #
    # print('weight of ALL spanning trees brute force: ', get_score_trees(all_spanning_trees, scores))
    #
    # # entry_point = torch.ones((mask_z_matrix.shape[0]), dtype=torch.int)
    # print('weight of ALL spanning trees MTT: ', get_score_trees_mtt(mask_z_matrix, scores))
    #
    # # tree weight brute force
    # # TODO
    #
    # # tree weight MTT theorem
    # # TODO

    # get_all_spanning_trees(mask_z_matrix)
