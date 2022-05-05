import json
import os

from cpn.tokenizer import TokenizerCPN
import pandas as pd

if __name__ == "__main__":
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/data-20200921'

    tokenizer_cpn = TokenizerCPN()
    nr_mentions = 0
    nr_linked_mentions = 0
    nr_docs = 0
    nr_clusters = 0
    nr_linked_clusters = 0
    nr_words = list()
    distinct_links = set()
    tags = {'all'}
    # tags = {'test'}
    # tags = {'train'}
    nr_docs_processed = 0
    prune_ratio = 0.2
    pruned_spans_sizes = list()


    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for curr_file in filenames:
            if 'DW_' in curr_file:
                linked_cluster_ids = set()
                cluster_ids = set()
                in_file_json = json.load(open(os.path.join(dirpath, curr_file), 'r'))
                curr_tags = set(in_file_json['tags'])
                intersec_tags = curr_tags.intersection(tags)
                if len(intersec_tags) == 0:
                    continue

                nr_docs_processed += 1
                if nr_docs_processed % 25 == 0:
                    print('nr of docs processed: ', nr_docs_processed)
                tokens = tokenizer_cpn.tokenize(in_file_json['content'])
                pruned_spans_sizes.append(int(len(tokens) * prune_ratio))
                nr_words.append(len(tokens))
                nr_docs += 1
                for curr_mention in in_file_json['mentions']:
                    nr_mentions += 1
                    curr_men_concept = in_file_json['concepts'][curr_mention['concept']]
                    if 'link' in curr_men_concept and curr_men_concept['link'] is not None and \
                            curr_men_concept['link'] != 'NILL':
                        # print('curr_men_concept link: ', curr_men_concept['link'])
                        distinct_links.add(curr_men_concept['link'])
                        nr_linked_mentions += 1
                        linked_cluster_ids.add(curr_mention['concept'])
                    cluster_ids.add(curr_mention['concept'])

                nr_linked_clusters += len(linked_cluster_ids)
                nr_clusters += len(cluster_ids)
                # print('in file json: ', in_file_json)

    print('nr words: ', sum(nr_words))
    print('nr mentions: ', nr_mentions)
    print('nr docs: ', nr_docs)
    print('avg doc length: ', sum(nr_words) / len(nr_words))
    print('nr linked mentions: ', nr_linked_mentions)
    print('nr linked clusters ', nr_linked_clusters)
    print('nr clusters ', nr_clusters)
    print('nr distinct links ', len(distinct_links))

    print('nr pruned spans ', pruned_spans_sizes)
    df = pd.DataFrame({'pruned_spans': pruned_spans_sizes, 'cases': 1})
    df_grouped = df.groupby(['pruned_spans']).count().reset_index()  # groupby(level=0).cumsum().reset_index()
    df_grouped['cumsum'] = df_grouped['cases'].cumsum()
    tot_clusters = df_grouped['cases'].sum()
    df_grouped['fraction_covered'] = df_grouped['cumsum'] / tot_clusters
    print('pandas dataframe: ')
    print(df_grouped.to_string(index=False))

