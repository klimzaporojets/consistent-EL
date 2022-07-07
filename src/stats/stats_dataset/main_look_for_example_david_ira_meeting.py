# Module that tries to find best example of mentions that refer to the same concept that are close to each other
# and the candidates of one (preferrably first one) do not include the correct candidate.
import json
import os

if __name__ == '__main__':
    # can be dwie or aida path
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/data-20200921'

    min_different_surface_forms_in_clusters = 2  # number of surface forms in the cluster
    max_mentions_in_cluster = 20  # maximum nr of mentions in cluster
    min_nr_clusters = 2  # with different_surface_forms_in_clusters differently written mentions

    # max length of the snippet
    snippet_max_length = 1000

    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for filename in filenames:
            if '.json' in filename and not 'main' in filename:
                curr_filename = filename
                parsed_json = json.load(open(os.path.join(dirpath, filename)))
                nr_mentions_no_correct_candidate = 0
                for idx, first_mention in enumerate(parsed_json['mentions']):
                    cluster_id_to_distinct_mentions_in_snippet = dict()
                    if 'candidates' not in first_mention:
                        continue
                    mentions_in_snippet = [first_mention]
                    concept_id = first_mention['concept']
                    if 'link' not in parsed_json['concepts'][concept_id] \
                            or parsed_json['concepts'][concept_id]['link'] is None:
                        continue

                    nr_same_found = 1
                    for curr_mention in parsed_json['mentions'][idx + 1:]:
                        if 'candidates' not in curr_mention:
                            continue

                        if curr_mention['begin'] - first_mention['begin'] > snippet_max_length:
                            break

                        if curr_mention['concept'] not in cluster_id_to_distinct_mentions_in_snippet:
                            cluster_id_to_distinct_mentions_in_snippet[curr_mention['concept']] = \
                                {'nr_mentions_cluster': 0,
                                 'mentions': set()}

                        cluster_id_to_distinct_mentions_in_snippet[curr_mention['concept']]['mentions'] \
                            .add(curr_mention['text'].lower())
                        cluster_id_to_distinct_mentions_in_snippet[curr_mention['concept']]['nr_mentions_cluster'] += 1

                    # sorts the found clusters by the number of distinct mentions in them

                    # sorted_clusters = sorted_clusters[0:min_nr_clusters]

                    cl_filtered = [cl for cl in cluster_id_to_distinct_mentions_in_snippet.values() if
                                   len(cl['mentions']) > min_different_surface_forms_in_clusters and
                                   cl['nr_mentions_cluster'] < max_mentions_in_cluster]

                    if len(cl_filtered) < min_nr_clusters:
                        continue
                    sorted_clusters = sorted(cluster_id_to_distinct_mentions_in_snippet.values(),
                                             key=lambda x: len(x['mentions']), reverse=True)

                    print('Snippet: ', parsed_json['content'][first_mention['begin']:curr_mention['end']])
                    print('clusters: ')
                    for curr_cluster in sorted_clusters:
                        print('Cluster mentions: ', curr_cluster['mentions'])
                        print('Mention count: ', curr_cluster['nr_mentions_cluster'])
