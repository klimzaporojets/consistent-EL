# The initial idea is to obtain the statistics on the maximum acc/f1 scores if only top x candidates are taken.
# Probably it would be nice to have a graph where on x axis is the nr of candidates and on y axis is the max possible
# acc/f1 score.
import json
import os

if __name__ == "__main__":
    dataset_path = 'data/data-20200921'
    # dataset_path = 'data/20200908-dataset-johannes'
    # for (dirpath, dirnames, filenames) in os.walk(path_results):

    # predicted - correct
    top_candidates = 16
    # top_candidates = 5
    account_for_nill = False
    candidates_lengths = []
    idxs_correct = []
    correct_all_docs = 0
    correct_all_concepts = 0
    correct_all_concepts_intersect = 0
    correct_prev_in_docs = 0
    correct_prev_in_docs_ent_level = 0
    correct_prev_in_docs_intersect_ent_level = 0
    correct_post_in_docs_ent_level = 0
    correct_post_in_docs = 0
    correct_1st_men_concept = 0

    tot_concepts = 0
    cpt_ok_all_doc = 0
    cpt_ok_all_concept = 0
    cpt_ok_all_concept_intersect = 0
    cpt_ok_1st_men_concept = 0

    heuristic_sort_length_text = False
    max_distinct_mention_candidates_per_document = 0

    nr_mentions_with_candidates = 0
    nr_mentions_with_correct_candidates = 0

    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for curr_file in filenames:
            if 'DW_' in curr_file:
                first_mention_concept_candidates = dict()
                prev_mentions_candidates = set()
                prev_mentions_candidates_ent_level = dict()  # concept level!!!!!
                prev_mentions_candidates_intersect_ent_level = dict()  # concept level!!!!!
                in_file_json = json.load(open(os.path.join(dirpath, curr_file), 'r'))
                all_candidates_in_file = set()
                all_candidates_per_concept = dict()
                all_candidates_intersect_per_concept = dict()

                if heuristic_sort_length_text:
                    in_file_json['mentions'] = sorted(in_file_json['mentions'], key=lambda x: len(x['text']),
                                                      reverse=True)

                for curr_mention in in_file_json['mentions']:
                    if 'candidates' in curr_mention:
                        if 'link' in in_file_json['concepts'][curr_mention['concept']] and \
                                in_file_json['concepts'][curr_mention['concept']]['link'] is not None and \
                                in_file_json['concepts'][curr_mention['concept']]['link'].lower() != 'nill':
                            nr_mentions_with_candidates += 1
                        candidates_with_scores = zip(curr_mention['candidates'], curr_mention['scores'])
                        candidates_with_scores = sorted(candidates_with_scores, key=lambda x: x[1],
                                                        reverse=True)
                        sorted_candidates = [cand[0] for cand in candidates_with_scores]
                        if top_candidates > -1:
                            sorted_candidates = sorted_candidates[:top_candidates]

                        if curr_mention['concept'] not in first_mention_concept_candidates:
                            first_mention_concept_candidates[curr_mention['concept']] = \
                                {curr_can for curr_can in sorted_candidates if curr_can is not None and
                                 curr_can != 'NILL'}
                            # first_mention_concept_candidates[curr_mention['concept']] = \
                            #     {curr_can for curr_can in curr_mention['candidates'] if curr_can is not None and
                            #      curr_can != 'NILL'}
                        for curr_candidate in sorted_candidates:
                            if 'link' in in_file_json['concepts'][curr_mention['concept']] and \
                                    in_file_json['concepts'][curr_mention['concept']]['link'] is not None:
                                if curr_candidate == in_file_json['concepts'][curr_mention['concept']]['link']:
                                    nr_mentions_with_correct_candidates += 1

                            # if idx_candidate >= top_candidates:
                            #     break
                            if curr_candidate is not None and curr_candidate != 'NILL':
                                all_candidates_in_file.add(curr_candidate)
                            if curr_mention['concept'] not in all_candidates_per_concept:
                                all_candidates_per_concept[curr_mention['concept']] = set()

                            all_candidates_per_concept[curr_mention['concept']].add(curr_candidate)

                        if curr_mention['concept'] not in all_candidates_intersect_per_concept:
                            all_candidates_intersect_per_concept[curr_mention['concept']] = set(sorted_candidates)

                        all_candidates_intersect_per_concept[curr_mention['concept']] = \
                            all_candidates_intersect_per_concept[curr_mention['concept']]. \
                                intersection(set(sorted_candidates))

                        if len(all_candidates_in_file) > max_distinct_mention_candidates_per_document:
                            max_distinct_mention_candidates_per_document = len(all_candidates_in_file)
                nr_mentions = len(in_file_json['mentions'])

                # if heuristic_sort_length_text:
                #     in_file_json['mentions'] = sorted(in_file_json['mentions'], key=lambda x: len(x['text']),
                #                                       reverse=True)

                for idx_curr_mention, curr_mention in enumerate(in_file_json['mentions']):
                    # ground_truth_link = None
                    curr_concept = in_file_json['concepts'][curr_mention['concept']]
                    # if 'link' in curr_concept and curr_concept['link'] is None:
                    #     ground_truth_link = 'NILL'

                    correct_idx = -1

                    if 'candidates' in curr_mention:
                        # if 'NILL' not in curr_mention['candidates'] and None not in curr_mention['candidates']:
                        #     curr_mention['candidates'].append('NILL')
                        #     curr_mention['scores'].append(-99999999)
                        candidates_lengths.append(len(curr_mention['candidates']))

                        # if 'link' not in curr_concept:
                        #     curr_concept['link'] = None
                        if 'link' in curr_concept:
                            if curr_concept['link'] is None or curr_concept['link'] == 'NILL':
                                if account_for_nill:
                                    correct_idx = 0
                                    idxs_correct.append(correct_idx)
                                    correct_all_docs += 1
                                else:
                                    continue
                            else:
                                correct_link = curr_concept['link']
                                candidates_with_scores = zip(curr_mention['candidates'], curr_mention['scores'])
                                candidates_with_scores = sorted(candidates_with_scores, key=lambda x: x[1],
                                                                reverse=True)

                                if top_candidates > -1:
                                    candidates_with_scores = candidates_with_scores[:top_candidates]
                                candidates_sorted = [cand[0] for cand in candidates_with_scores]

                                for curr_cand in candidates_sorted:
                                    if curr_cand is not None and curr_cand != 'NILL':
                                        prev_mentions_candidates.add(curr_cand)
                                        if curr_mention['concept'] not in prev_mentions_candidates_ent_level:
                                            prev_mentions_candidates_ent_level[curr_mention['concept']] = set()

                                        prev_mentions_candidates_ent_level[curr_mention['concept']].add(curr_cand)

                                if curr_mention['concept'] not in prev_mentions_candidates_intersect_ent_level:
                                    prev_mentions_candidates_intersect_ent_level[curr_mention['concept']] = set()

                                if len(prev_mentions_candidates_intersect_ent_level[curr_mention['concept']]) == 0:
                                    prev_mentions_candidates_intersect_ent_level[curr_mention['concept']] = \
                                        set(candidates_sorted)
                                else:
                                    prev_mentions_candidates_intersect_ent_level[curr_mention['concept']] = \
                                        prev_mentions_candidates_intersect_ent_level[curr_mention['concept']] \
                                            .intersection(candidates_sorted)

                                post_mens_cands_ent_level = dict()  # concept level!!!!!
                                post_mentions_candidates = set()
                                if idx_curr_mention < nr_mentions:
                                    for curr_post_mention in in_file_json['mentions'][idx_curr_mention:]:
                                        if 'candidates' in curr_post_mention:
                                            candidates_post_with_scores = zip(curr_post_mention['candidates'],
                                                                              curr_post_mention['scores'])
                                            candidates_post_with_scores = sorted(candidates_post_with_scores,
                                                                                 key=lambda x: x[1],
                                                                                 reverse=True)

                                            if top_candidates > -1:
                                                candidates_post_with_scores = candidates_post_with_scores[
                                                                              :top_candidates]
                                            candidates_post_sorted = [cand[0] for cand in candidates_post_with_scores]

                                            for curr_post_candidate in candidates_post_sorted:
                                                if curr_post_mention is not None and curr_post_candidate != 'NILL':
                                                    post_mentions_candidates.add(curr_post_candidate)
                                                    if curr_post_mention['concept'] not in post_mens_cands_ent_level:
                                                        #######
                                                        post_mens_cands_ent_level[curr_post_mention['concept']] = set()
                                                    post_mens_cands_ent_level[curr_post_mention['concept']].add(
                                                        curr_post_candidate)

                                correct_idx = candidates_sorted.index(
                                    correct_link) if correct_link in candidates_sorted else -1

                                idxs_correct.append(correct_idx)
                                curr_mention_concept = curr_mention['concept']
                                if curr_mention_concept in all_candidates_per_concept:
                                    if correct_link in all_candidates_per_concept[curr_mention_concept]:
                                        correct_all_concepts += 1

                                if correct_link in all_candidates_intersect_per_concept[curr_mention_concept]:
                                    correct_all_concepts_intersect += 1

                                if correct_link in all_candidates_in_file:
                                    correct_all_docs += 1

                                if correct_link in prev_mentions_candidates:
                                    correct_prev_in_docs += 1

                                if curr_mention_concept in prev_mentions_candidates_ent_level:
                                    if correct_link in prev_mentions_candidates_ent_level[curr_mention_concept]:
                                        correct_prev_in_docs_ent_level += 1

                                if correct_link in prev_mentions_candidates_intersect_ent_level[curr_mention_concept]:
                                    correct_prev_in_docs_intersect_ent_level += 1

                                if correct_link in post_mentions_candidates:
                                    correct_post_in_docs += 1

                                if curr_mention_concept in post_mens_cands_ent_level:
                                    if correct_link in post_mens_cands_ent_level[curr_mention_concept]:
                                        correct_post_in_docs_ent_level += 1

                                if curr_mention_concept in first_mention_concept_candidates:
                                    if correct_link in first_mention_concept_candidates[curr_mention_concept]:
                                        correct_1st_men_concept += 1

                for curr_concept in in_file_json['concepts']:
                    if 'link' in curr_concept and curr_concept['link'] is not None and curr_concept['link'] != 'NILL':
                        correct_link = curr_concept['link']
                        if curr_concept['concept'] in all_candidates_intersect_per_concept:
                            tot_concepts += 1
                            if correct_link in all_candidates_intersect_per_concept[curr_concept['concept']]:
                                cpt_ok_all_concept_intersect += 1

                        if curr_concept['concept'] in all_candidates_per_concept:
                            if correct_link in all_candidates_per_concept[curr_concept['concept']]:
                                cpt_ok_all_concept += 1
                        if correct_link in all_candidates_in_file:
                            cpt_ok_all_doc += 1

                        if curr_concept['concept'] in first_mention_concept_candidates:
                            if correct_link in first_mention_concept_candidates[curr_concept['concept']]:
                                cpt_ok_1st_men_concept += 1

    print('fraction mentions with correct candidate in its candidate list: ',
          nr_mentions_with_correct_candidates / nr_mentions_with_candidates)
    # print('The candidate lengths are: ', candidates_lengths)
    print('Avg nr of candidates: ', sum(candidates_lengths) / len(candidates_lengths))
    print('Max nr of candidates: ', max(candidates_lengths))

    print('total of mentions considered for correct idxs: ', len(idxs_correct))
    elems_upper_bound = [idxc for idxc in idxs_correct if idxc > -1]

    print('total of mentions considered for correct idxs: ', len(elems_upper_bound),
          len(elems_upper_bound) / len(idxs_correct))

    print('total of mentions considered for correct idxs (on concept-level): ', correct_all_concepts,
          correct_all_concepts / len(idxs_correct))

    print('total of mentions considered for correct idxs (intersection of candidates on concept-level): ',
          correct_all_concepts_intersect, correct_all_concepts_intersect / len(idxs_correct))

    print('total of mentions considered for correct idxs (all doc): ', correct_all_docs,
          correct_all_docs / len(idxs_correct))

    print('total of mentions considered for correct idxs (prev mentions): ', correct_prev_in_docs,
          correct_prev_in_docs / len(idxs_correct))

    print('total of mentions considered for correct idxs (prev mentions concept level): ',
          correct_prev_in_docs_ent_level,
          correct_prev_in_docs_ent_level / len(idxs_correct))

    print('total of mentions considered for correct idxs (post mentions): ', correct_post_in_docs,
          correct_post_in_docs / len(idxs_correct))

    print('total of mentions considered for correct idxs (post mentions concept level): ',
          correct_post_in_docs_ent_level, correct_post_in_docs_ent_level / len(idxs_correct))

    print('total of mentions considered where first mention of concept has right candidate: ',
          correct_1st_men_concept, correct_1st_men_concept / len(idxs_correct))

    print('total of mentions considered for correct idxs (intersect of candidates of prev mentions concept level): ',
          correct_prev_in_docs_intersect_ent_level,
          correct_prev_in_docs_intersect_ent_level / len(idxs_correct))

    print('maximum nr of candidates per file: ', max_distinct_mention_candidates_per_document)

    print('1st ok concept: ', cpt_ok_1st_men_concept / tot_concepts)
    print('all ok concept: ', cpt_ok_all_concept_intersect / tot_concepts)
    print('>= 1 ent-level ok: ', cpt_ok_all_concept / tot_concepts)
    print('>= 1 doc-level ok: ', cpt_ok_all_doc / tot_concepts)
