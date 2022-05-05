import json

from datass.dictionary import Dictionary

if __name__ == "__main__":
    linking_dict_path = "/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/" \
                        "data-20200921-bert/links_dictionary.json"
    problematic_file = "/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/" \
                       "data-20200921-bert/dwie_bert_s384/DW_446861.json"

    dictionary = Dictionary()
    dictionary.load_json(linking_dict_path)

    parsed_file = json.load(open(problematic_file, 'rt'))

    problematic_mention = 'Johanna Spyri'

    problematic_mention_spans = set()

    problematic_mention_idxs = []

    problematic_mention_cands = []

    problematic_mention_corr_cands = []

    problematic_mention_concept = None


    for curr_mention in parsed_file['mentions']:
        if curr_mention['text'] == problematic_mention:
            problematic_mention_spans.add((curr_mention['subtoken_begin'], curr_mention['subtoken_end']))
            problematic_mention_concept = curr_mention['concept']

    for idx, curr_span in enumerate(parsed_file['all_spans']):
        curr_span  = tuple(curr_span)
        if curr_span in problematic_mention_spans:
            problematic_mention_idxs.append(idx)

    for curr_men_idx in problematic_mention_idxs:
        print('candidates for ', curr_men_idx, ': ')
        print([tk for tk in parsed_file['all_spans_candidates'][curr_men_idx]])
        print([dictionary.get(tk) for tk in parsed_file['all_spans_candidates'][curr_men_idx]])
        target_idx = parsed_file['all_spans_candidates_target'][curr_men_idx]
        print('correct target: ', curr_men_idx, ': ',
              dictionary.get(parsed_file['all_spans_candidates'][curr_men_idx][target_idx]))
