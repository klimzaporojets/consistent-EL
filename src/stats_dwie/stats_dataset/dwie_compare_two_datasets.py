# just compares two datasets the original motive is to compare the DWIE version Severine (MS student) is working,
# and the one publicly available online
import json
import os

if __name__ == "__main__":
    paths_compare = [
        # '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_public/data/annos',
        '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/uplifting_annotations/annotations/export/output',
        '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/mt2021_severineverlinden/dataset/severine']

    dataset_stats = [
        {'nr_mentions': 0,
         'nr_concepts': 0,
         'nr_concepts_mentions': 0,
         'nr_relations': 0,
         'nr_candidates': 0,
         # 'nr_rel_types': 0,
         'nr_ner_types': 0,
         'nr_ner_types_total': 0},
        {'nr_mentions': 0,
         'nr_concepts': 0,
         'nr_concepts_mentions': 0,
         'nr_relations': 0,
         'nr_candidates': 0,
         # 'nr_rel_types': 0,
         'nr_ner_types': 0,
         'nr_ner_types_total': 0}
    ]

    for idx, curr_path_compare in enumerate(paths_compare):
        tot_ner_types = set()
        for (dirpath, dirnames, filenames) in os.walk(curr_path_compare):
            for filename in filenames:
                if 'DW_' in filename:
                    infile_path = os.path.join(curr_path_compare, filename)
                    print('processing file: ', infile_path)
                    with open(infile_path) as infile:
                        parsed_file = json.load(infile)
                        dataset_stats[idx]['nr_mentions'] += len(parsed_file['mentions'])
                        dataset_stats[idx]['nr_relations'] += len(parsed_file['relations'])
                        dataset_stats[idx]['nr_concepts'] += len(parsed_file['concepts'])
                        concepts_mentions = set()
                        for curr_mention in parsed_file['mentions']:
                            concepts_mentions.add(curr_mention['concept'])
                            if 'candidates' in curr_mention:
                                dataset_stats[idx]['nr_candidates'] += len(curr_mention['candidates'])
                        dataset_stats[idx]['nr_concepts_mentions'] += len(concepts_mentions)

                        tot_nr_ner_tags = 0
                        for curr_concept in parsed_file['concepts']:
                            tot_nr_ner_tags += len(curr_concept['tags'])
                            tot_ner_types = tot_ner_types.union(set(curr_concept['tags']))

                        dataset_stats[idx]['nr_ner_types'] += tot_nr_ner_tags

                        dataset_stats[idx]['nr_ner_types_total'] = len(tot_ner_types)

    # finds the specific documents with concept ids where the ner annotations differ
    nr_mentions_different_pos = 0
    curr_path_compare = paths_compare[0]
    for (dirpath, dirnames, filenames) in os.walk(curr_path_compare):
        for filename in filenames:
            if 'DW_' in filename:
                infile_path1 = os.path.join(curr_path_compare, filename)
                infile_path2 = os.path.join(paths_compare[1], filename)
                with open(infile_path1) as infile:
                    parsed_file1 = json.load(infile)

                with open(infile_path2) as infile:
                    parsed_file2 = json.load(infile)

                for curr_concept1 in parsed_file1['concepts']:
                    curr_concept2 = parsed_file2['concepts'][curr_concept1['concept']]
                    if len(curr_concept1['tags']) != len(curr_concept2['tags']):
                        print('difference for concept, ', curr_concept1['concept'], ' (', curr_concept1['text'],
                              ') in file: ', filename)
                        print(curr_concept1['tags'], ' vs ', curr_concept2['tags'])

                for idx_mention, curr_mention1 in enumerate(parsed_file1['mentions']):
                    curr_mention2 = parsed_file2['mentions'][idx_mention]
                    if curr_mention1['begin'] != curr_mention2['begin'] or \
                            curr_mention1['end'] != curr_mention2['end']:
                        nr_mentions_different_pos += 1

    print('dataset stats: ', dataset_stats)
    print('nr of mentions with different pos: ', nr_mentions_different_pos)
