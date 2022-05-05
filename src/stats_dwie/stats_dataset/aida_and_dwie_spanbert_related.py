import json
import os

import numpy as np
import pandas
import pandas as pd

if __name__ == "__main__":
    # span_bert_dir = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/' \
    #                 'data-20200921-bert/dwie_bert_s384'
    span_bert_dir = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida/' \
                    'aida_reannotated/aida-20210402/transformed/spanbert_s384'

    max_span_length = 0
    subtoken_width_to_nr_of_mentions = dict()
    number_of_mentions = []
    number_of_pruned_spans = []
    nr_loaded = 0
    tok_mention_ratio = []

    for (dirpath, dirnames, filenames) in os.walk(span_bert_dir):
        for filename in filenames:
            # if 'DW_' in filename:
            if 'links_dictionary' not in filename and 'json' in filename:
                loaded = json.load(open(os.path.join(dirpath, filename)))
                all_spans = loaded['all_spans']
                for curr_span in all_spans:
                    span_length = curr_span[1] - curr_span[0] + 1
                    if span_length > max_span_length:
                        max_span_length = span_length
                nr_loaded += 1
                if nr_loaded % 100 == 0:
                    print('nr of loaded: ', nr_loaded)
                number_of_mentions.append(len(loaded['mentions']))
                number_of_pruned_spans.append(int(len(loaded['begin_token']) * 0.45))
                for curr_mention in loaded['mentions']:
                    subtoken_begin = curr_mention['subtoken_begin']
                    subtoken_end = curr_mention['subtoken_end']
                    subtoken_width = subtoken_end - subtoken_begin + 1
                    if subtoken_width not in subtoken_width_to_nr_of_mentions:
                        subtoken_width_to_nr_of_mentions[subtoken_width] = 0
                    subtoken_width_to_nr_of_mentions[subtoken_width] += 1

                tok_mention_ratio.append((len(loaded['mentions'])) / len(loaded['begin_token']))

    df_nr_mentions = pd.DataFrame({'mentions': number_of_mentions, 'nr_mentions': number_of_mentions})
    df_nr_mentions_g = df_nr_mentions.groupby('mentions').count().reset_index()
    # df_span_widths_g.rename(columns={df_span_widths_g.columns[1]: 'span_width'}, inplace=True)
    df_nr_mentions_g.sort_values(by=['mentions'], ascending=True, inplace=True)
    df_nr_mentions_g['nr_mentions_cum'] = df_nr_mentions_g['nr_mentions'].cumsum()
    max_cum = df_nr_mentions_g.max(skipna=False)['nr_mentions_cum']
    df_nr_mentions_g['nr_mentions_cum_perc'] = (df_nr_mentions_g['nr_mentions_cum'] / max_cum) * 100
    print('==========NR OF MENTIONS========')
    print(df_nr_mentions_g)
    df_nr_pruned_spans = pd.DataFrame(
        {'pruned_spans': number_of_pruned_spans, 'nr_pruned_spans': number_of_pruned_spans})
    df_nr_pruned_spans_g = df_nr_pruned_spans.groupby('pruned_spans').count().reset_index()
    # df_span_widths_g.rename(columns={df_span_widths_g.columns[1]: 'span_width'}, inplace=True)
    df_nr_pruned_spans_g.sort_values(by=['pruned_spans'], ascending=True, inplace=True)
    df_nr_pruned_spans_g['nr_pruned_spans_cum'] = df_nr_pruned_spans_g['nr_pruned_spans'].cumsum()
    max_cum = df_nr_pruned_spans_g.max(skipna=False)['nr_pruned_spans_cum']
    df_nr_pruned_spans_g['nr_pruned_spans_cum_perc'] = (df_nr_pruned_spans_g['nr_pruned_spans_cum'] / max_cum) * 100
    print('==========NR OF PRUNED SPANS========')
    print(df_nr_pruned_spans_g)

    tot_nr_of_mentions = sum(subtoken_width_to_nr_of_mentions.values())
    sumcum = 0
    for i in range(max(subtoken_width_to_nr_of_mentions.keys())):
        if (i + 1) in subtoken_width_to_nr_of_mentions:
            sumcum += subtoken_width_to_nr_of_mentions[i + 1]
            print('width ', i + 1, ' sumcum: ', sumcum, ' and sumcum %: ', (sumcum / tot_nr_of_mentions) * 100,
                  '   out of ', tot_nr_of_mentions)
    print('max_span_length: ', max_span_length)

    print('======= nr tokens / nr mentions ratio========')
    df_tok_men_ratio = pd.DataFrame({'tok_men_ratio': tok_mention_ratio,
                                     'nr_tok_men_ratio': [1 / len(tok_mention_ratio)] * len(tok_mention_ratio)})

    df_tok_men_ratio_g = df_tok_men_ratio.groupby(
        pd.cut(df_tok_men_ratio['tok_men_ratio'], np.arange(0, 1.0 + 0.01, 0.01))).sum()
    df_tok_men_ratio_g = df_tok_men_ratio_g.drop(columns='tok_men_ratio').reset_index()
    df_tok_men_ratio_g['nr_tok_men_ratio_cum'] = df_tok_men_ratio_g['nr_tok_men_ratio'].cumsum()
    df_tok_men_ratio_g['right_interval'] = [d.right for d in df_tok_men_ratio_g['tok_men_ratio']]
    df_tok_men_ratio_g = df_tok_men_ratio_g.drop(columns=['nr_tok_men_ratio', 'tok_men_ratio'])
    pandas.set_option('display.max_rows', df_tok_men_ratio_g.shape[0] + 1)

    print(df_tok_men_ratio_g)
