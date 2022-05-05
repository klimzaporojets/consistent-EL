# THE GOAL OF THIS MODULE IS TO ANNOTATE COREFERENCE IN AIDA DOCUMENTS
import os


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


def complete_with_cluster_id(ask_cluster_id_lines, ask_cluster_id_toks, assigned_cluster_ids, ner_type, ner_types):
    if len(assigned_cluster_ids) > 0:
        max_cluster_id = max(assigned_cluster_ids) + 1
    else:
        max_cluster_id = 0
    assert len(ask_cluster_id_lines) == len(ner_types)
    to_ask = ' '.join(ask_cluster_id_toks) + '({} - new cluster: {})'.format(ner_type, max_cluster_id)
    cluster_id = input(to_ask)
    cluster_id = int(cluster_id)
    assigned_cluster_ids.append(cluster_id)
    to_ret_lines = ['{}\t{}\t{}\n'.format(curr_l.strip(), ner_types[idx], cluster_id)
                    for idx, curr_l in enumerate(ask_cluster_id_lines)]
    return to_ret_lines


def annotate_tmp_lines(lines_aida, lines_conll, wikilink_to_cluster_id, wikilink_to_mentions):
    assert len(lines_aida) == len(lines_conll)
    to_ret_lines = []
    nil_nr = 0
    caught_nil = False
    ask_cluster_id_toks = []
    ask_cluster_id_lines = []
    # assigned_cluster_ids = []
    assigned_cluster_ids = list(wikilink_to_cluster_id.values())
    cluster_id_to_wikilink = dict((v, k) for k, v in wikilink_to_cluster_id.items())
    # wikilink_to_cluster_id = dict()
    ner_type = ''
    ner_types = list()
    print('The following are cluster ids already used for wikilinks: ')
    for curr_cluster_id in assigned_cluster_ids:
        wiki_link = cluster_id_to_wikilink[curr_cluster_id]
        mentions = [' '.join(currm) for currm in wikilink_to_mentions[wiki_link]]
        mentions = '", "'.join(mentions)
        print('{}-{}-({})'.format(curr_cluster_id, wiki_link, mentions))
    for curr_aida_line, curr_conll_line in zip(lines_aida, lines_conll):
        spl_line_aida = curr_aida_line.split('\t')
        spl_line_conll = curr_conll_line.split(' ')
        if len(spl_line_aida) >= 4:
            # hard check consistency AIDA vs CONLL 2003 (ALL LINKS ARE NERS)
            # assert len(spl_line_conll) >= 4 and spl_line_conll[3].strip() in CONLL_NER_TAGS

            wikilink = spl_line_aida[3].strip()
            bio_tag = spl_line_aida[1]

            if bio_tag == 'B' and caught_nil:
                assert ner_type != ''
                ask_cluster_id_toks.append("###NL_E({})###".format(nil_nr))
                ask_cluster_id_lines = complete_with_cluster_id(ask_cluster_id_lines, ask_cluster_id_toks,
                                                                assigned_cluster_ids, ner_type, ner_types)
                to_ret_lines.extend(ask_cluster_id_lines)
                ask_cluster_id_lines = []
                ask_cluster_id_toks = []
                nil_nr += 1
                caught_nil = False
            if bio_tag == 'B':
                ner_types = list()

            ner_type = spl_line_conll[3].strip()
            ner_types.append(ner_type)

            if bio_tag == 'B' or bio_tag == 'I':
                # exist_link_in_this_file = True
                if wikilink == '--NME--':
                    if not caught_nil:
                        ask_cluster_id_toks.append("###NL_B({})###".format(nil_nr))
                    ask_cluster_id_toks.append(spl_line_aida[0])
                    ask_cluster_id_lines.append(curr_aida_line)
                    caught_nil = True
                else:
                    assert not caught_nil
                    if wikilink not in wikilink_to_cluster_id:
                        if len(assigned_cluster_ids) > 0:
                            wikilink_to_cluster_id[wikilink] = max(assigned_cluster_ids) + 1
                        else:
                            wikilink_to_cluster_id[wikilink] = 0
                        assigned_cluster_ids.append(wikilink_to_cluster_id[wikilink])

                    cluster_id = wikilink_to_cluster_id[wikilink]
                    to_ret_lines_app = '{}\t{}\t{}\n'.format(curr_aida_line.strip(), ner_type, cluster_id)
                    # print(to_ret_lines_app)
                    to_ret_lines.append(to_ret_lines_app)
        else:
            if len(ask_cluster_id_lines) > 0:
                assert caught_nil
                assert ner_type != ''
                assert len(ner_types) == len(ask_cluster_id_lines)
                ask_cluster_id_toks.append("###NL_E({})###".format(nil_nr))
                ask_cluster_id_lines = complete_with_cluster_id(ask_cluster_id_lines, ask_cluster_id_toks,
                                                                assigned_cluster_ids, ner_type, ner_types)
                to_ret_lines.extend(ask_cluster_id_lines)
                ask_cluster_id_lines = []
                ask_cluster_id_toks = []
                caught_nil = False
                nil_nr += 1
                ner_types = []
            to_ret_lines.append(curr_aida_line)

    if len(ask_cluster_id_lines) > 0:
        assert caught_nil
        assert ner_type != ''
        assert len(ner_types) == len(ask_cluster_id_lines)

        ask_cluster_id_toks.append("###NL_E({})###".format(nil_nr))
        ask_cluster_id_lines = complete_with_cluster_id(ask_cluster_id_lines, ask_cluster_id_toks,
                                                        assigned_cluster_ids, ner_type, ner_types)
        to_ret_lines.extend(ask_cluster_id_lines)
        # ask_cluster_id_lines = []
        # ask_cluster_id_toks = []
        # caught_nil = False
        # nil_nr += 1

    # id_cluster = input("Type something to test this out: ")
    # print('the cluster is ', id_cluster)
    return to_ret_lines


if __name__ == "__main__":
    CONLL_NER_TAGS_B = {'B-LOC', 'B-MISC', 'B-ORG', 'B-PER'}
    CONLL_NER_TAGS_I = {'I-LOC', 'I-MISC', 'I-ORG', 'I-PER'}
    CONLL_NER_TAGS = {'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER'}

    dataset_paths = [{'tag': 'testa', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                                   'dwie_linker/data/aida/aida_testa.tsv',
                      'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
                                    'eng.testa'},
                     {'tag': 'train', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                                   'dwie_linker/data/aida/aida_train.tsv',
                      'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
                                    'eng.train'},
                     {'tag': 'testb', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/'
                                                   'dwie_linker/data/aida/aida_testb.tsv',
                      'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
                                    'eng.testb'}]
    # dataset_paths = [{'tag': 'train', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                                'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                                'aida_train.tsv',
    #                   'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
    #                                 'eng.train'},
    #                  {'tag': 'testa', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                                'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                                'aida_testa.tsv',
    #                   'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
    #                                 'eng.testa'},
    #                  {'tag': 'testb', 'path_aida': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/'
    #                                                'entities/aida-yago2-dataset/aida-yago2-dataset/partitioned/'
    #                                                'aida_testb.tsv',
    #                   'path_conll': '/home/ibcn044/work_files/ugent/phd_work/datasets/cpn_related/entities/conll2003/'
    #                                 'eng.testb'}]
    for curr_dataset_path in dataset_paths:
        aida_yago_path = curr_dataset_path['path_aida']
        aida_coref_path = aida_yago_path + '.coref.tsv'
        conll_path = curr_dataset_path['path_conll']
        tag = curr_dataset_path['tag']
        content_curr_aida = []
        content_curr_conll = []
        already_annotated_aida_coref = []

        with open(aida_yago_path) as infile:
            for aida_line in infile:
                content_curr_aida.append(aida_line)

        with open(conll_path) as infile:
            for aida_line in infile:
                content_curr_conll.append(aida_line)

        if os.path.isfile(aida_coref_path):
            with open(aida_coref_path) as infile:
                for aida_coref_line in infile:
                    already_annotated_aida_coref.append(aida_coref_line)

        curr_tag = curr_dataset_path['tag']
        # with open(aida_yago_path) as infile:
        new_doc = False
        doc_tokens = []
        caught_nil = False
        caught_nnil = False
        tmp_lines_aida = []
        tmp_lines_conll = []

        nil_nr = 0
        idx_conll = 0
        wikilnk_to_cluster_id = dict()
        wikilnk_to_mentions = dict()
        wikilnk_mentions = list()
        wikilink = ''

        positioned = False
        # idx_anno_aida_coref = 0
        for idx_aida, aida_line in enumerate(content_curr_aida):
            spl_line_aida = aida_line.split('\t')
            if not positioned and idx_aida < len(already_annotated_aida_coref):
                # checks integrity of annotations, the tokens have to match!!, if not we have a problem
                assert spl_line_aida[0].strip() == already_annotated_aida_coref[idx_aida].split('\t')[0].strip()
                if len(spl_line_aida) >= 4:
                    # the wiki link has to also match!
                    assert spl_line_aida[3].strip() == already_annotated_aida_coref[idx_aida].split('\t')[3].strip()

                # if the length is 1 in original file, in annotated as well, we only annotate...
                continue
            elif not positioned and idx_aida == len(already_annotated_aida_coref):
                positioned = True
            # automatically positions on the last saved document
            curr_token_aida = spl_line_aida[0].strip()
            conll_line = content_curr_conll[idx_conll]
            spl_line_conll = conll_line.split(' ')
            curr_token_conll = spl_line_conll[0].strip()
            if '-DOCSTART-' in curr_token_conll:
                curr_token_conll = '-DOCSTART-'
            if '-DOCSTART-' in curr_token_aida:
                curr_token_aida = '-DOCSTART-'

            while curr_token_aida != curr_token_conll:
                idx_conll += 1
                conll_line = content_curr_conll[idx_conll]
                spl_line_conll = conll_line.split(' ')
                curr_token_conll = spl_line_conll[0].strip()
                if '-DOCSTART-' in curr_token_conll:
                    curr_token_conll = '-DOCSTART-'

            assert curr_token_aida == curr_token_conll
            # print(curr_token_aida, ' vs. ', curr_token_conll)
            # print(tag, ': idx conll: ', idx_conll, ' idx aida: ', idx_aida)

            idx_conll += 1
            if new_doc:
                doc_tokens = []
                new_doc = False

            if '-DOCSTART-' in aida_line:
                print(' '.join(doc_tokens))
                if len(tmp_lines_aida) > 0:
                    tmp_lines_anno = annotate_tmp_lines(tmp_lines_aida, tmp_lines_conll, wikilnk_to_cluster_id,
                                                        wikilnk_to_mentions)
                    # print()
                    what_to_do = input(
                        'DO YOU WANT TO PERSIST THE CHANGES FOR THIS DOCUMENT (Y), OR JUST FORGET ABOUT IT '
                        'AND EXIT (N)?')
                    what_to_do = what_to_do.lower().strip()
                    # if what_to_do == 'n':
                    if what_to_do != 'y':
                        exit(0)
                    with open(aida_coref_path, 'a+') as app_file:
                        for curr_line in tmp_lines_anno:
                            app_file.write(curr_line)

                tmp_lines_aida = []
                tmp_lines_conll = []
                tmp_lines_aida.append(aida_line)
                tmp_lines_conll.append(conll_line)
                new_doc = True
                nil_nr = 0
                wikilnk_to_cluster_id = dict()
                wikilnk_to_mentions = dict()
                wikilnk_mentions = list()
                wikilink = ''
            else:
                tmp_lines_aida.append(aida_line)
                tmp_lines_conll.append(conll_line)

                if len(spl_line_aida) >= 4:
                    # hard check consistency AIDA vs CONLL 2003 (ALL LINKS ARE NERS)
                    assert len(spl_line_conll) >= 4 and spl_line_conll[3].strip() in CONLL_NER_TAGS

                    bio_tag = spl_line_aida[1]
                    if bio_tag == 'B' and caught_nnil:
                        caught_nnil = False
                        if wikilink not in wikilnk_to_mentions:
                            wikilnk_to_mentions[wikilink] = list()
                        wikilnk_to_mentions[wikilink].append(wikilnk_mentions)
                        wikilnk_mentions = list()
                    wikilink = spl_line_aida[3].strip()
                    if bio_tag == 'B' and caught_nil:
                        doc_tokens.append("###NL_E({})###".format(nil_nr))
                        nil_nr += 1
                        caught_nil = False
                    if bio_tag == 'B' or bio_tag == 'I':
                        exist_link_in_this_file = True
                        if wikilink == '--NME--':
                            if not caught_nil:
                                doc_tokens.append("###NL_B({})###".format(nil_nr))
                            caught_nil = True
                        else:
                            assert caught_nil == False
                            caught_nnil = True
                            if wikilink not in wikilnk_to_cluster_id:
                                if len(wikilnk_to_cluster_id) == 0:
                                    wiki_cluster_id = 0
                                else:
                                    wiki_cluster_id = max(wikilnk_to_cluster_id.values()) + 1
                                wikilnk_to_cluster_id[wikilink] = wiki_cluster_id
                            wikilnk_mentions.append(spl_line_aida[0])
                else:
                    if len(spl_line_conll) >= 4 and spl_line_conll[3].strip() in CONLL_NER_TAGS:
                        print('WARNNNNN!!!: CHECKING NILS IS NOT ENOUGH!!!', spl_line_conll, idx_conll)
                        # hard check consistency AIDA vs CONLL 2003 (ALL NERS ANNOTATED WITH LINKS)
                        raise Exception('WARNNNNN!!!: CHECKING NILS IS NOT ENOUGH!!!')
                    if caught_nil:
                        doc_tokens.append("###NL_E({})###".format(nil_nr))
                        nil_nr += 1
                        # print('---------------------------------')
                        caught_nil = False

                    if caught_nnil:
                        caught_nnil = False
                        assert wikilink != ''
                        if wikilink not in wikilnk_to_mentions:
                            wikilnk_to_mentions[wikilink] = list()
                        wikilnk_to_mentions[wikilink].append(wikilnk_mentions)
                        wikilnk_mentions = list()

                doc_tokens.append(spl_line_aida[0].strip())

        if len(tmp_lines_aida) > 0:
            tmp_lines_anno = annotate_tmp_lines(tmp_lines_aida, tmp_lines_conll, wikilnk_to_cluster_id,
                                                wikilnk_to_mentions)
            # print()
            what_to_do = input(
                'DO YOU WANT TO PERSIST THE CHANGES FOR THIS DOCUMENT (Y), OR JUST FORGET ABOUT IT '
                'AND EXIT (N)?')
            what_to_do = what_to_do.lower().strip()
            # if what_to_do == 'n':
            if what_to_do != 'y':
                exit(0)
            with open(aida_coref_path, 'a+') as app_file:
                for curr_line in tmp_lines_anno:
                    app_file.write(curr_line)
