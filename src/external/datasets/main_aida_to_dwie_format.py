# the goal of this module is to transform documents formatted in AIDA CoNLL-YAGO format into the DWIE (CPN) format

def join_tokens(doc_tokens_begin, doc_tokens_end, doc_tokens):
    to_ret = ''
    assert len(doc_tokens_begin) == len(doc_tokens_end)
    assert len(doc_tokens_begin) == len(doc_tokens)
    last_token_end = 0
    for curr_token, curr_begin, curr_end in zip(doc_tokens, doc_tokens_begin, doc_tokens_end):
        # if last_token_end > 0:
        #     assert curr_begin > last_token_end
        assert curr_begin - last_token_end <= 1  # no more than one space allowed

        to_ret += ' ' * (curr_begin - last_token_end)
        to_ret += curr_token
        last_token_end = curr_end
    return to_ret


if __name__ == "__main__":
    # dataset_paths = [{'tag': 'testa', 'path': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                           'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                           'aida_testa.tsv'},
    #                  {'tag': 'train', 'path': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                           'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                           'aida_train.tsv'},
    #                  {'tag': 'testb', 'path': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                           'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                           'aida_testb.tsv'}]
    dataset_paths = [{'tag': 'testa', 'path': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                              'dwie_linker/data/aida/aida_testa.tsv'},
                     {'tag': 'train', 'path': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                              'dwie_linker/data/aida/aida_train.tsv'},
                     {'tag': 'testb', 'path': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                              'dwie_linker/data/aida/aida_testb.tsv'}]
    for curr_dataset_path in dataset_paths:
        aida_yago_path = curr_dataset_path['path']
        curr_tag = curr_dataset_path['tag']
        with open(aida_yago_path) as infile:
            new_doc = False
            doc_tokens_begin = []
            doc_tokens_end = []
            doc_tokens = []
            doc_content = ''
            doc_mentions = []
            doc_clusters = []

            prev_token = None
            curr_begin = 0
            spaced = False
            quote_started = False
            for line in infile:
                if new_doc:
                    spaced = True  # we do not want spaces in the beginning of the document
                    doc_tokens_begin = []
                    doc_tokens_end = []
                    doc_tokens = []
                    doc_content = ''
                    curr_begin = 0
                    new_doc = False
                    quote_started = False
                if '-DOCSTART-' in line:
                    new_doc = True
                    prev_token = None
                    curr_doc_content = join_tokens(doc_tokens_begin, doc_tokens_end, doc_tokens)
                    print('========================================')
                    print('CURR CONTENT IS: ')
                    print(curr_doc_content)
                    print('========================================')
                else:
                    line = line.split('\t')
                    curr_token = line[0].strip()
                    # if prev_token is not None and prev_token != '':
                    #     if prev_token in ',.' and not spaced:
                    #         curr_begin += 1
                    #         spaced = True
                    if (curr_token not in ',.)') \
                            and (prev_token is not None and prev_token not in '(') \
                            and (not curr_token.startswith('\'')) \
                            and (curr_token != 'n\'t') \
                            and (curr_token != '...') \
                            and (curr_token != ':') \
                            and (curr_token != '?') \
                            and (curr_token != ';') \
                            and (curr_token != '/') \
                            and (prev_token != '$') \
                            and (prev_token != '/') \
                            and (prev_token != '\"' or (not quote_started)) \
                            and (curr_token != '\"' or (not quote_started)) \
                            and not spaced:
                        curr_begin += 1
                        spaced = True

                    if curr_token == '' and not spaced:
                        curr_begin += 1
                        spaced = True

                    elif curr_token != '':
                        spaced = False
                        curr_end = curr_begin + len(curr_token)
                        doc_tokens.append(curr_token)
                        doc_tokens_begin.append(curr_begin)
                        doc_tokens_end.append(curr_end)
                        curr_begin = curr_end

                    if curr_token == '':
                        # has to be added, we do not want to lose the original tokenization given in AIDA
                        doc_tokens.append(curr_token)
                        doc_tokens_begin.append(curr_begin)
                        doc_tokens_end.append(curr_begin)

                    if curr_token == '\"' and not quote_started:
                        quote_started = True
                    elif curr_token == '\"' and quote_started:
                        quote_started = False
                    prev_token = curr_token
