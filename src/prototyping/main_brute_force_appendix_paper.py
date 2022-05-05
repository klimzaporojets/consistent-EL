# the goal is to reproduce the calculation for the appendix in the paper
# Also see figure/diagramas in https://app.diagrams.net/#G1hxaab1zbsPwvg2bPnAsEuhF4tE7_UvBW
# I am basing myself on main_brute_force_mtt.py module developed before.
import itertools
import math

import torch

from prototyping.main_brute_force_mtt import get_all_spanning_trees_v2, get_score_trees, get_score_trees_mtt, \
    get_nr_trees_brute_restrictive_v2, get_scores_brute_restrictive_v2, get_score_trees_mtt_partition

if __name__ == "__main__":
    nr_elements = 5
    mask_z_matrix = torch.IntTensor([[0, 1, 1, 1, 1, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 1, 1, 0]])

    mask_target_matrix = torch.IntTensor([[0, 1, 1, 1, 0, 1],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0]])

    # scores = torch.FloatTensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
    #                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    #                             [0.0, 0.0, 0.0, 0.0, 5.0, 6.0],
    #                             [0.0, 0.0, 0.0, 0.0, -5.0, 7.0],
    #                             [0.0, 0.0, 0.0, -4.0, 0.0, -2.0],
    #                             [0.0, 0.0, 0.0, 4.0, -3.0, 0.0]])
    scores = torch.FloatTensor([[0.0, 1.0, 1.0, 5.0, 3.0, 7.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 4.0, 2.0],
                                [0.0, 0.0, 0.0, 0.0, 5.0, 9.0],
                                [0.0, 0.0, 0.0, 3.0, 0.0, 2.0],
                                [0.0, 0.0, 0.0, 8.0, 4.0, 0.0]])

    # I do not take exp here because for the example in the paper we directly work in exp space
    # scores = torch.exp(scores)

    all_spanning_trees = get_all_spanning_trees_v2(nr_elements + 1, mask_z_matrix)
    target_spanning_trees = get_all_spanning_trees_v2(nr_elements + 1, mask_target_matrix)

    score_brute_all = get_score_trees(all_spanning_trees, scores)
    score_mtt_all = get_score_trees_mtt(mask_z_matrix, scores)

    print('score of ALL spanning trees brute force: ', score_brute_all, ' and in log space: ',
          math.log(score_brute_all))
    print('score of ALL spanning trees MTT: ', score_mtt_all.item())

    clusters = [[3, 5], [2, 4]]
    link_idx = {1, 2}
    nr_connection_br_force = get_nr_trees_brute_restrictive_v2(target_spanning_trees, clusters)
    w_connection_br_force = get_scores_brute_restrictive_v2(target_spanning_trees, scores, clusters,
                                                            link_idxs=link_idx)
    print('nr of TARGET spanning trees w. single connection to cluster brute force: ', nr_connection_br_force)
    print('score of TARGET spanning trees w. single connection from root to cluster for each cluster (brute force): ',
          w_connection_br_force, ' and in log space: ', math.log(w_connection_br_force))

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
        # tot_nil_scores_single_conn *= curr_score
        # kzaporoj (04/11/2021) - I think it has to be += since it is in log space, so I commented *= (prev line)
        # and added next line with +=
        tot_nil_scores_single_conn += curr_score

    print('NIL clusters experimental weight using linear partition in log space: ', tot_nil_scores_single_conn)
    # print('NIL clusters experimental weight using linear partition approach: ', tot_nil_scores_single_conn,
    #       ' and in log space: ', math.log(tot_nil_scores_single_conn))
