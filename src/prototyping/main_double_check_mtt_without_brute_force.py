# similar to main_brute_force_mtt.py but for real big matrices, using mtt only, so no brute forcing here
import json
import os
import sys

import torch

from prototyping.main_brute_force_mtt import get_score_trees_mtt, get_score_trees_mtt_partition

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_dir = sys.argv[1]
    else:
        file_dir = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker_mtt_check/' \
                   'matrices_20210514e/debugging_logs/'

    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        loaded_json = json.load(open(file_path, 'rt'))

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
        # nr_elements = len(flat_mention_ids)
        nr_elements = len(items) - 1

        # print('nr of elements: ', scores.shape[-1] - 1)  # -1 because of root

        mask_target_matrix = torch.zeros_like(scores, dtype=torch.int)
        mask_z_matrix = torch.zeros_like(scores, dtype=torch.int)
        mask_z_matrix[0, 1:] = 1  # from root can reach anywhere
        not_nil_idxs_in_matrix_calculated = set()
        not_nil_idxs_in_matrix_calculated.add(0)  # root
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
                            if (el1_cluster, el2_cluster) in possible_link_spans_combinations:
                                mask_target_matrix[el1_cluster, el2_cluster] = 1
            if not link_in_cluster:
                idx_torch_cluster = torch.tensor(curr_cluster, dtype=torch.long)
                mask_target_matrix[0, :][idx_torch_cluster] = 1  # 1 from root to all mentions in cluster ??
            else:
                not_nil_idxs_in_matrix_calculated = not_nil_idxs_in_matrix_calculated.union(set(curr_cluster))
        for curr_link_idx in link_idx:
            # all the links have to be connected to root by definition in target
            mask_target_matrix[0, curr_link_idx] = 1
            not_nil_idxs_in_matrix_calculated.add(curr_link_idx)

        not_nil_idxs_in_matrix_passed = loaded_json['numer_explained']['not_nil_idxs_in_matrix']
        # not_nil_idxs_in_matrix_calculated = None # TODO
        not_nil_idxs_in_matrix_calculated = sorted(list(not_nil_idxs_in_matrix_calculated))
        diff1 = set(not_nil_idxs_in_matrix_passed).difference(set(not_nil_idxs_in_matrix_calculated))
        diff2 = set(not_nil_idxs_in_matrix_calculated).difference(set(not_nil_idxs_in_matrix_passed))

        if len(diff1) > 0:
            print('difference not nil idxs passed 1: ', diff1)
            print([items[cit] for cit in diff1])
            print([cit for cit in diff1])
        if len(diff2) > 0:
            print('difference not nil idxs passed 2: ', diff2)
            print([items[cit] for cit in diff2])
            print([cit for cit in diff2])

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
            mask_z_matrix[curr_link_span_comb[0], curr_link_span_comb[1]] = 1

        # just assigns the scores higher than -inf there

        # mask_z_matrix = (scores > -10000000000.0).int()

        if 'target_mask' in loaded_json and 'z_mask' in loaded_json:
            saved_target_mask = torch.IntTensor(loaded_json['target_mask'])[0]
            saved_z_mask = torch.IntTensor(loaded_json['z_mask'])[0]
            are_target_masks_equal = torch.all(mask_target_matrix.eq(saved_target_mask))
            # print('ARE target masks equal: ', are_target_masks_equal)
            assert are_target_masks_equal
            are_z_masks_equal = torch.all(mask_z_matrix.eq(saved_z_mask))
            # print('ARE z masks equal: ', are_z_masks_equal)
            assert are_z_masks_equal

        masked_tgt_scores = scores * mask_target_matrix
        masked_z_scores = scores * mask_z_matrix

        scores = torch.arcsinh(scores)
        scores = torch.log(mask_z_matrix.float()) + scores
        scores = torch.exp(scores)

        # print('file was parsed')

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
        assert len(nil_clusters_indices) == len(set(nil_clusters_indices))
        nil_clusters_indices = [0] + nil_clusters_indices
        nil_clusters_indices = torch.tensor(nil_clusters_indices, dtype=torch.long)
        nr_iters = 0
        tot_nil_scores_single_conn = 0.0

        assert len(nil_clusters) == len(loaded_json['numer_explained']['nil_clusters_weights'])
        # now trying to do it using mtt more efficiently for each cluster once (no need of cartesian product)
        for idx_c_cl, curr_nil_cluster in enumerate(nil_clusters):
            ind_root = torch.tensor(curr_nil_cluster, dtype=torch.long)
            # ind_root = ind_root + 1  # accounting for the first root column
            # appends also the root
            ind_root = torch.cat([torch.tensor([0], dtype=torch.long), ind_root])
            scores_cluster = scores[ind_root, :][:, ind_root]
            mask_target_matrix_c = mask_target_matrix.clone().detach()
            mask_target_matrix_c[0, :] = 0
            mask_target_matrix_c[0, ind_root] = 1
            mask_target_matrix_c = mask_target_matrix_c[ind_root, :][:, ind_root]
            mask_target_matrix_c[0][0] = 0
            curr_score = get_score_trees_mtt_partition(mask_target_matrix_c, scores_cluster)
            tot_nil_scores_single_conn += curr_score

        # print('NIL clusters experimental weight using linear partition approach: ', tot_nil_scores_single_conn,
        #       ' and in log space: ', math.log(tot_nil_scores_single_conn))

        tot_not_nil_scores = torch.Tensor([0.0])
        link_idxs_added = set()
        for curr_not_nil_cluster in not_nil_clusters:
            for cidx in curr_not_nil_cluster:
                if cidx in link_idx:
                    link_idxs_added.add(cidx)
            ind_cluster = torch.tensor([0] + curr_not_nil_cluster, dtype=torch.long)
            curr_mask_target = mask_target_matrix[ind_cluster, :][:, ind_cluster]
            curr_scores = scores[ind_cluster, :][:, ind_cluster]
            curr_score = get_score_trees_mtt(curr_mask_target, curr_scores)
            # print('not nil cluster score log space: ', curr_score.item())
            if tot_not_nil_scores == 0.0:
                tot_not_nil_scores = curr_score
            else:
                tot_not_nil_scores = tot_not_nil_scores + curr_score

        # BEGIN: adds the rest of the pointers from the root to links
        link_idxs_to_add = link_idx.difference(link_idxs_added)
        ind_cluster = torch.tensor([0] + list(link_idxs_to_add), dtype=torch.long)
        curr_mask_target = mask_target_matrix[ind_cluster, :][:, ind_cluster]
        curr_scores = scores[ind_cluster, :][:, ind_cluster]
        curr_score = get_score_trees_mtt(curr_mask_target, curr_scores)
        # print('not nil cluster score log space: ', curr_score.item())
        if tot_not_nil_scores == 0.0:
            tot_not_nil_scores = curr_score
        else:
            tot_not_nil_scores = tot_not_nil_scores + curr_score
        # END: adds the rest of the pointers from the root to links

        score_mtt_all = get_score_trees_mtt(mask_z_matrix, scores)

        tot_score = tot_not_nil_scores + tot_nil_scores_single_conn

        # print('nr nil clusters of us: ', len(nil_clusters))
        # print('nr nil clusters weights predicted: ', len(loaded_json['numer_explained']['nil_clusters_weights']))

        # print('double checking for repeated elements in clusters: ')
        clusters_gold = loaded_json['clusters_gold']
        clusters_gold_list = [item for sublist in clusters_gold for item in sublist]
        assert len(clusters_gold_list) == len(set(clusters_gold_list))
        seen_spans_pred = set()
        for curr_cluster_pred in loaded_json['clusters_pred_all']:
            for curr_span_in_cluster_pred in seen_spans_pred:
                if curr_span_in_cluster_pred in seen_spans_pred:
                    print('something wrong with this cluster:', curr_cluster_pred,
                          ', element seen: ', curr_span_in_cluster_pred)
                seen_spans_pred.add(curr_span_in_cluster_pred)
        clusters_pred_all = loaded_json['clusters_pred_all']
        clusters_pred_all_list = [item for sublist in clusters_pred_all for item in sublist]
        assert len(clusters_pred_all_list) == len(set(clusters_pred_all_list))

        print('======================================')
        print('processing ', file_path)
        print('---')
        print('tot denom score: ', score_mtt_all.item())
        print('tot denom score (predicted): ', loaded_json['denom'])
        print('difference denom: ', score_mtt_all.item() - loaded_json['denom'])
        print('---')

        print('tot numer score: ', tot_score.item())
        print('tot numer score (predicted): ', loaded_json['numer'])
        print('difference numer: ', tot_score.item() - loaded_json['numer'])
        print('---')

        print('tot not nil scores: ', tot_not_nil_scores.item())
        print('tot not nil weights (predicted): ', loaded_json['numer_explained']['not_nil_weight'])
        print('difference tot nil scores: ',
              (tot_not_nil_scores.item() - loaded_json['numer_explained']['not_nil_weight']))
        print('---')

        print('tot nil scores: ', tot_nil_scores_single_conn)
        print('tot nil clusters weights (predicted): ', sum(loaded_json['numer_explained']['nil_clusters_weights']))
        print('difference tot nil scores: ',
              (tot_nil_scores_single_conn - sum(loaded_json['numer_explained']['nil_clusters_weights'])).item())
