# Module that tries to find best example of mentions that refer to the same concept that are close to each other
# and the candidates of one (preferrably first one) do not include the correct candidate.
import json
import os

if __name__ == "__main__":
    # can be dwie or aida path
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/data-20200921'
    # dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida/aida_reannotated/aida-20210402/transformed/aida_tokenization_adaptation'
    nr_consecutive_mentions = 2  # lowest distance between 3 mentions referring to the same concept

    # max length of the snippet
    snippet_max_length = 120

    max_candidates_to_consider = 2
    first_mention_no_correct_candidate = False
    min_nr_mentions_no_correct_candidate = 1
    max_nr_correct_candidates = 1

    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for filename in filenames:
            if '.json' in filename and not 'main' in filename:
                curr_filename = filename
                parsed_json = json.load(open(os.path.join(dirpath, filename)))
                nr_mentions_no_correct_candidate = 0
                for idx, first_mention in enumerate(parsed_json['mentions']):
                    nr_correct_found = 0
                    found_correct = False
                    if 'candidates' not in first_mention:
                        continue
                    mentions_in_snippet = [first_mention]
                    concept_id = first_mention['concept']
                    if 'link' not in parsed_json['concepts'][concept_id] \
                            or parsed_json['concepts'][concept_id]['link'] is None:
                        continue

                    gold_link = parsed_json['concepts'][concept_id]['link']
                    if first_mention_no_correct_candidate and \
                            gold_link in first_mention['candidates'][:max_candidates_to_consider]:
                        continue
                    elif not first_mention_no_correct_candidate and \
                            gold_link not in first_mention['candidates'][:max_candidates_to_consider]:
                        continue
                    elif gold_link in first_mention['candidates'][:max_candidates_to_consider]:
                        found_correct = True
                        nr_correct_found += 1
                    nr_same_found = 1
                    for curr_mention in parsed_json['mentions'][idx + 1:]:
                        if 'candidates' not in curr_mention:
                            continue
                        if curr_mention['concept'] == concept_id:
                            nr_same_found += 1
                            if gold_link in curr_mention['candidates'][:max_candidates_to_consider]:
                                found_correct = True
                                nr_correct_found += 1
                            mentions_in_snippet.append(curr_mention)
                            distance_from_first = curr_mention['begin'] - first_mention['begin']
                            if distance_from_first > snippet_max_length:
                                break

                            if nr_same_found == nr_consecutive_mentions:
                                if not found_correct:
                                    break
                                if nr_correct_found > max_nr_correct_candidates:
                                    break
                                print('=' * 100)
                                print(curr_filename)
                                print('Snippet ground truth: ', gold_link)
                                from_snippet = max([0, first_mention['begin'] - 20])
                                to_snippet = min([len(parsed_json['content']), curr_mention['end'] + 20])
                                print('Snippet length: ', distance_from_first)
                                print('Snippet: ', parsed_json['content'][from_snippet: to_snippet])
                                print('Top candidates (top ', max_candidates_to_consider, ')')
                                for men_snippet in mentions_in_snippet:
                                    print(men_snippet['text'], ': ',
                                          men_snippet['candidates'])
