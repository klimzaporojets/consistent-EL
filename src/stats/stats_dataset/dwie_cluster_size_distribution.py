import json
import os
import pandas

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/data-20200921'
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida/aida_reannotated/aida/output-json/current/'
    concept_sizes = list()
    nr_mentions_in_doc = list()
    nr_clusters_in_doc = list()
    nr_nil_cluster_cartesian_mtt = list()

    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for curr_file in filenames:
            # if 'DW_' in curr_file:
            if '.json' in curr_file and not 'main' in curr_file:

                nill_cluster_cartesian = 0
                nr_nil_clusters = 0
                doc_cluster_ids = set()
                in_file_json = json.load(open(os.path.join(dirpath, curr_file), 'r'))
                concept_to_men_nr = dict()
                for curr_mention in in_file_json['mentions']:
                    if curr_mention['concept'] not in concept_to_men_nr:
                        concept_to_men_nr[curr_mention['concept']] = 0
                    concept_to_men_nr[curr_mention['concept']] += 1
                    doc_cluster_ids.add(curr_mention['concept'])
                concept_sizes.extend(list(concept_to_men_nr.values()))

                nr_mentions_in_doc.append(len(in_file_json['mentions']))

                nr_clusters_in_doc.append(len(doc_cluster_ids))
                for curr_cluster_id in doc_cluster_ids:
                    cluster_data = in_file_json['concepts'][curr_cluster_id]
                    if 'link' not in cluster_data or cluster_data['link'] is None:
                        cluster_size = cluster_data['count']
                        nr_nil_clusters += 1
                        if nill_cluster_cartesian == 0:
                            nill_cluster_cartesian = cluster_size
                        else:
                            nill_cluster_cartesian = nill_cluster_cartesian * cluster_size
                if 'train' in in_file_json['tags']:
                    nr_nil_cluster_cartesian_mtt.append({'doc_id': in_file_json['id'],
                                                     'cartesian_size': nill_cluster_cartesian,
                                                     'nr_nil_clusters': nr_nil_clusters})
                else:
                    nr_nil_cluster_cartesian_mtt.append({'doc_id': in_file_json['id'],
                                                     'cartesian_size': nill_cluster_cartesian,
                                                     'nr_nil_clusters': nr_nil_clusters})

    df = pd.DataFrame(nr_nil_cluster_cartesian_mtt)
    df = df.sort_values(['cartesian_size'], ascending=[0])
    df = df.reset_index().reset_index()

    print('cartesian size: ')
    pandas.set_option('display.max_rows', df.shape[0] + 1)
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    bars = df.plot(ax=ax, y='cartesian_size',
                       # color=color, label='',
                        x='level_0',
                       title='Carteian size plot')
    plt.show()
    print('==========')
    # cluster size statistic
    df = pd.DataFrame({'concept_size': concept_sizes, 'concepts': 1})
    df_grouped = df.groupby(['concept_size']).count().reset_index()  # groupby(level=0).cumsum().reset_index()
    df_grouped['cumsum'] = df_grouped['concepts'].cumsum()
    tot_clusters = df_grouped['concepts'].sum()
    df_grouped['fraction_covered'] = df_grouped['cumsum'] / tot_clusters
    print('pandas dataframe: ')
    print(df_grouped.to_string(index=False))

    # mentions in doc statistic
    df = pd.DataFrame({'mentions_in_doc': nr_mentions_in_doc, 'docs': 1})
    df_grouped = df.groupby(['mentions_in_doc']).count().reset_index()  # groupby(level=0).cumsum().reset_index()
    df_grouped['cumsum'] = df_grouped['docs'].cumsum()
    tot_clusters = df_grouped['docs'].sum()
    df_grouped['fraction_covered'] = df_grouped['cumsum'] / tot_clusters
    print('pandas dataframe: ')
    print(df_grouped.to_string(index=False))

    # clusters in doc statistic
    df = pd.DataFrame({'clusters_in_doc': nr_clusters_in_doc, 'docs': 1})
    df_grouped = df.groupby(['clusters_in_doc']).count().reset_index()  # groupby(level=0).cumsum().reset_index()
    df_grouped['cumsum'] = df_grouped['docs'].cumsum()
    tot_clusters = df_grouped['docs'].sum()
    df_grouped['fraction_covered'] = df_grouped['cumsum'] / tot_clusters
    print('pandas dataframe: ')
    print(df_grouped.to_string(index=False))
