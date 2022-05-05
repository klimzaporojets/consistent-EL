# the initial goal of this module is to see why in the logs the coverage of entity embeddings for aida seems to be
# low (around 76.5%) from logs:
# shape of all_embeddings:  (201787, 200)
# found: 201787 / 263803 = 0.7649154861771853.
# The idea is to load both the dictionary and embedding files to understand why there is a mismatch
# it can be that some links are unicode-encoded and have to be decoded first for example; or any other reason.
import itertools
import json
import os

if __name__ == "__main__":
    link_embeddings_file = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/' \
                           'entity_embeddings/johannes_yamada_31082020/enwiki_200.txt'
    link_embeddings_file_serialized = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/' \
                                      'entity_embeddings/johannes_yamada_31082020/enwiki_200_serial.json'
    aida_dictionary_file = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida/' \
                           'aida_reannotated/aida-20210402/transformed/spanbert_s384/links_dictionary.json'
    links_in_embeddings = set()
    links_in_dictionary = dict()

    aida_dictionary = json.load(open(aida_dictionary_file, 'rt'))
    nr_added = 0
    if not os.path.exists(link_embeddings_file_serialized):
        for line in open(link_embeddings_file, 'rt'):
            if not line.startswith('1#'):
                nr_added += 1
                if nr_added % 100000 == 0:
                    print('nr added links: ', nr_added)
                    # break
                link_to_add = line[:line.index(' ')].strip()
                links_in_embeddings.add(link_to_add)
                # print('one line of embedding file is: ', line)
        to_serialize = list(sorted(links_in_embeddings))
        json.dump(to_serialize, open(link_embeddings_file_serialized, 'w'))
    else:
        links_in_embeddings = json.load(open(link_embeddings_file_serialized, 'rt'))
        links_in_embeddings = set(links_in_embeddings)

    found = 0
    total = 0
    not_found_list = list()
    do_fixed1 = True
    not_found_fixed1_list = list()
    print('length of aida_dictionary: ', len(aida_dictionary))
    for k, v in aida_dictionary.items():
        total += 1
        if total % 10000 == 0:
            print('processed of aida dictionary: ', total)
        if k in links_in_embeddings:
            found += 1
        else:
            is_fixed = False
            # tries to combine first letter lowercased
            if do_fixed1:
                sp_k = k.split('_')
                # print('splitted length: ', len(sp_k))
                if len(sp_k) <= 10:
                    # lst_combinations = list()
                    for curr_comb in itertools.product([0, 1], repeat=len(sp_k)):
                        # print('curr comb: ', curr_comb)
                        to_search = ''
                        for idx_c, comb in enumerate(curr_comb):
                            curr_token = sp_k[idx_c]
                            if comb == 1 and curr_token[0].isalpha():
                                curr_token = curr_token[0].upper() + curr_token[1:]
                            if comb == 0 and curr_token[0].isalpha():
                                curr_token = curr_token[0].lower() + curr_token[1:]
                            if idx_c == 0:
                                to_search = curr_token
                            else:
                                to_search = to_search + '_' + curr_token
                        if to_search in links_in_embeddings:
                            is_fixed = True
                            not_found_fixed1_list.append({'original': k, 'fixed': to_search})
                            found += 1
                            break
            if is_fixed:
                continue
            not_found_list.append(k)
    print('total: ', total, ' found: ', found, ' % found: ', (found / total) * 100)
    print('fixed 1: ', len(not_found_fixed1_list))
    print('the dictionary was parsed')
    # TODOs: lowercasing each of the components of the wikipedia url.
    # Academic_boycotts_of_Israel --> Academic_boycott_of_Israel (in embedding file)
