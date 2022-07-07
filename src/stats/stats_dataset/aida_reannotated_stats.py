# stats of aida reannotated, comparing it with the aida original dataset
import json
import os


def load_raw_file(aida_paths):
    aida_id_to_raw_doc = dict()
    for curr_aida_path in aida_paths:
        with open(curr_aida_path, 'rt') as infile:
            for curr_line in infile:
                curr_line_json = json.loads(curr_line)
                aida_id_to_raw_doc[curr_line_json['id']] = curr_line_json
    return aida_id_to_raw_doc


def load_raw_dir_path(aida_path):
    aida_id_to_raw_doc = dict()
    for (dirpath, dirnames, filenames) in os.walk(aida_path):
        curr_file_name: str = None
        for curr_file_name in filenames:
            if curr_file_name.endswith('.json'):
                loaded_raw_json = json.load(open(os.path.join(dirpath, curr_file_name), 'rt'))
                aida_id_to_raw_doc[loaded_raw_json['id']] = loaded_raw_json
    return aida_id_to_raw_doc


def get_stats_doc(parsed_doc):
    nr_linked_mentions = 0
    nr_nil_mentions = 0
    nr_nil_clusters = 0
    nr_not_nil_clusters = 0
    processed_clusters = set()
    for curr_mention in parsed_doc['mentions']:
        mention_concept_id = curr_mention['concept']
        curr_mention_cluster = parsed_doc['concepts'][mention_concept_id]
        if 'link' in curr_mention_cluster and curr_mention_cluster['link'] is not None \
                and curr_mention_cluster['link'].lower() != 'nill':
            if mention_concept_id not in processed_clusters:
                nr_not_nil_clusters += 1
                processed_clusters.add(mention_concept_id)
            nr_linked_mentions += 1
        else:
            if mention_concept_id not in processed_clusters:
                nr_nil_clusters += 1
                processed_clusters.add(mention_concept_id)
            nr_nil_mentions += 1
    return {'nr_linked_mentions': nr_linked_mentions,
            'nr_nil_mentions': nr_nil_mentions,
            'nr_linked_clusters': nr_not_nil_clusters,
            'nr_nil_clusters': nr_nil_clusters}


# def get_stats_from_raw(raw_aida_orig, raw_aida_reannotated):
def get_stats_from_raw(to_get_stats):
    nr_linked_mentions = dict()
    nr_nil_mentions = dict()
    nr_linked_clusters = dict()
    nr_nil_clusters = dict()

    for curr_dataset in to_get_stats:
        dataset_tag = curr_dataset['tag']
        parsed = curr_dataset['parsed']
        nr_linked_mentions[dataset_tag] = 0
        nr_nil_mentions[dataset_tag] = 0
        nr_linked_clusters[dataset_tag] = 0
        nr_nil_clusters[dataset_tag] = 0

        for curr_doc_id, curr_doc in parsed.items():
            curr_doc_stats = get_stats_doc(curr_doc)
            nr_linked_mentions[dataset_tag] += curr_doc_stats['nr_linked_mentions']
            nr_nil_mentions[dataset_tag] += curr_doc_stats['nr_nil_mentions']
            nr_linked_clusters[dataset_tag] += curr_doc_stats['nr_linked_clusters']
            nr_nil_clusters[dataset_tag] += curr_doc_stats['nr_nil_clusters']

        # prints in latex format
        print('\t {} & {:,} & {:,} & {:,} & {:,} \\\\ '.format(dataset_tag,
                                                               nr_linked_clusters[dataset_tag],
                                                               nr_nil_clusters[dataset_tag],
                                                               nr_linked_mentions[dataset_tag],
                                                               nr_nil_mentions[dataset_tag]))


if __name__ == "__main__":
    aida_original_path_testa = 'data/aida/aida/partitioned/aida_dwie_format_no_coref_annos/testa.jsonl'
    aida_original_path_testb = 'data/aida/aida/partitioned/aida_dwie_format_no_coref_annos/testb.jsonl'
    aida_original_path_train = 'data/aida/aida/partitioned/aida_dwie_format_no_coref_annos/train.jsonl'
    aida_reannotated_path = 'data/aida/aida_reannotated/aida-20210402/transformed/aida_tokenization_adaptation/'

    dwie_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_public/data/annos_with_content'

    aida_original_paths = [aida_original_path_train, aida_original_path_testa, aida_original_path_testb]

    raw_aida_orig = load_raw_file(aida_original_paths)
    raw_aida_reannotated = load_raw_dir_path(aida_reannotated_path)
    raw_dwie = load_raw_dir_path(dwie_path)

    to_get_stats = [{'tag': '\\dwiedataset', 'parsed': raw_dwie},
                    {'tag': '\\originalaida', 'parsed': raw_aida_orig},
                    {'tag': '\\ouraida', 'parsed': raw_aida_reannotated}]

    # get_stats_from_raw(raw_aida_orig, raw_aida_reannotated)
    get_stats_from_raw(to_get_stats)

    nr_linked_mentions_reannotated = None  # TODO
    nr_linked_mentions_original = None

    clusters_per_cluster_size_reanno = None
    clusters_per_cluster_size_orig = None
