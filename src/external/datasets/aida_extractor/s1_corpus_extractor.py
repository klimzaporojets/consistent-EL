import json
import os
import struct

from typing import Set, List, Dict


def save_dwie_in_jsonl(dwie_formatted: List, file_path: str):
    with open(file_path, 'wt') as outfile:
        for curr_json in dwie_formatted:
            outfile.write(json.dumps(curr_json) + '\n')


def save_dwie_in_json_files(dwie_formatted: List, dir_path: str):
    for curr_json in dwie_formatted:
        file_name = '{}.json'.format(curr_json['id'])
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'wt') as outfile:
            json.dump(curr_json, outfile, indent=4)


def get_next_article(file):
    # to_ret_id = None
    to_ret_content = None
    version = file.read(4)
    if len(version) == 0:
        return None, None
    # print('current version is ', version)
    unpacked_version = struct.unpack('>i', version)

    assert unpacked_version[0] == -3
    fields = struct.unpack('>i', file.read(4))
    # print('nr of fields: ', fields[0])
    to_ret_fields = dict()
    for i in range(fields[0]):
        # print('processing field ', i)
        utf_length = struct.unpack('>H', file.read(2))[0]
        fieldName = file.read(utf_length)
        fieldName = fieldName.decode('utf-8')
        # print('read fieldName: ', fieldName)
        curr_type = struct.unpack('>b', file.read(1))
        curr_type = curr_type[0]
        fieldValue = None
        if curr_type == 0:
            # print('curr type 0')
            size = struct.unpack('>i', file.read(4))[0]
            fieldValue = file.read(size)
            fieldValue = fieldValue.decode('utf-8')
            # print('fieldValue is: ', fieldValue)
        elif curr_type == 1:
            # print('curr type 1')
            raise Exception('I don\'t know (yet) how to read type 1!')
        to_ret_fields[fieldName] = fieldValue

    assert 'identifier' in to_ret_fields
    assert 'content' in to_ret_fields

    to_ret_id = to_ret_fields['identifier']
    to_ret_content = to_ret_fields['content']
    return to_ret_id, to_ret_content


def load_in_list(file_path, separator):
    to_ret_lst = list()
    if os.path.isfile(file_path):
        with open(file_path) as infile:
            for curr_line in infile:
                splt_line = curr_line.split(separator)
                splt_line = [sl.strip() for sl in splt_line]
                to_ret_lst.append(splt_line)
    return to_ret_lst


# def get_next_not_empty_line(lst_content: List, pointer):
def get_next_not_empty_line(lst_content: List):
    to_ret_line = None
    while len(lst_content) > 0:
        curr_content = lst_content.pop(0)
        if curr_content[0] != '':
            to_ret_line = curr_content
            break
    return to_ret_line


def get_next_ne_line_file(infile, separator):
    to_ret_line = None
    for curr_content in infile:
        curr_content = parse_tsv(curr_content, separator)
        if curr_content[0] != '':
            to_ret_line = curr_content
            break
    return to_ret_line


def get_next_id(existent_ids: Set, default=0):
    if len(existent_ids) == 0:
        existent_ids.add(default)
        return default
    else:
        next_id = max(existent_ids) + 1
        existent_ids.add(next_id)
        return next_id


def add_curr_mention(crr_mention, crr_concept_id, cpt_id_to_concept_details, cpt_id_to_mentions):
    if crr_concept_id not in cpt_id_to_concept_details:
        cpt_id_to_concept_details[crr_concept_id] = {'id': crr_concept_id,
                                                     'text': crr_mention['text'],
                                                     'type': {crr_mention['type']},
                                                     'link': set()}

    if crr_mention['link'] is not None:
        cpt_id_to_concept_details[crr_concept_id]['link'].add(crr_mention['link'])

    assert crr_mention['type'] is not None
    cpt_id_to_concept_details[crr_concept_id]['type'].add(crr_mention['type'])

    if crr_concept_id not in cpt_id_to_mentions:
        cpt_id_to_mentions[crr_concept_id] = list()

    cpt_id_to_mentions[crr_concept_id].append(crr_mention)

    return concept_id_to_concept_details, cpt_id_to_mentions


def parse_tsv(content, separator):
    content = content.split(separator)
    content = [cnt.strip() for cnt in content]
    return content


def regroup_by_link_and_ner_type(cpt_id_to_mentions: Dict, cpt_id_to_cpt_details: Dict, concept_id, all_cpts_ids: Set):
    config_to_cpt_id = dict()

    # print('passed mentions to re-group: ', mentions)
    mentions = cpt_id_to_mentions[concept_id]
    cpt_id_to_mentions[concept_id] = list()
    # cpt_id_to_cpt_details[concept_id] = None
    del cpt_id_to_cpt_details[concept_id]
    print('=============================================================')
    print('regrouping: ', mentions)
    for curr_mention in mentions:
        if curr_mention['link'] is not None:
            curr_type = 'link'
        else:
            curr_type = curr_mention['type']
        if curr_type not in config_to_cpt_id:
            if len(config_to_cpt_id) == 0:
                config_to_cpt_id[curr_type] = curr_mention['concept']
            else:
                new_cpt_id = get_next_id(all_cpts_ids)
                config_to_cpt_id[curr_type] = new_cpt_id
                curr_mention['concept'] = new_cpt_id
        else:
            curr_mention['concept'] = config_to_cpt_id[curr_type]

        cpt_id_to_cpt_details, cpt_id_to_mentions = \
            add_curr_mention(curr_mention, curr_mention['concept'], cpt_id_to_cpt_details, cpt_id_to_mentions)
    if len(config_to_cpt_id) > 1:
        print('regrouped WITH changes: ', mentions)
    else:
        print('regrouped without changes: ', mentions)
    print('=============================================================')


def get_in_dwie_format(curr_mentions, cpt_id_to_mentions, cpt_id_to_cpt_details, tokenization,
                       doc_tag, doc_id, doc_raw_content, break_multiple_ner_types=True):
    # if break_multiple_ner_types is in True, only breaks the ones coreferenced through manual annotation
    # (e.g. by using complete_corefs.py annotation), but not the ones originally coreferenced through linking

    assert len(cpt_id_to_mentions) == len(cpt_id_to_cpt_details)
    cnt_cpt_mens = sum([len(cl) for cl in cpt_id_to_mentions.values()])
    assert len(curr_mentions) == cnt_cpt_mens

    # id = "AIDA_{}".format(doc_id)
    id = "{}".format(doc_id)
    to_ret_dwie_format = dict()
    to_ret_dwie_format['id'] = id
    to_ret_dwie_format['content'] = doc_raw_content
    to_ret_dwie_format['tokenization'] = dict()
    assert len(tokenization['tokens']) == len(tokenization['begin'])
    assert len(tokenization['begin']) == len(tokenization['end'])
    assert len(tokenization['end']) == len(tokenization['sentences'])
    
    to_ret_dwie_format['tokenization']['tokens'] = tokenization['tokens']
    to_ret_dwie_format['tokenization']['begin'] = tokenization['begin']
    to_ret_dwie_format['tokenization']['end'] = tokenization['end']
    to_ret_dwie_format['tokenization']['sentences'] = tokenization['sentences']
    to_ret_dwie_format['tags'] = list()
    to_ret_dwie_format['tags'].append('all')
    to_ret_dwie_format['tags'].append(doc_tag)

    if break_multiple_ner_types:
        nr_concepts_old = len(cpt_id_to_mentions)
        nr_mentions_old = len(curr_mentions)

        all_concept_ids = set(cpt_id_to_mentions.keys())
        for curr_concept_id in range(len(cpt_id_to_mentions)):
            curr_concept_details = cpt_id_to_cpt_details[curr_concept_id]
            if len(curr_concept_details['type']) >= 2 and break_multiple_ner_types:
                # if this check fails, then we have another problem...
                # assert len(curr_concept_details['link']) > 0
                # re-groups just by link first, and then those without link groups by ner type
                # inside changes the 'concept' field in the mention
                regroup_by_link_and_ner_type(cpt_id_to_mentions, cpt_id_to_cpt_details, curr_concept_id,
                                             all_concept_ids)

        # just in case checks that the nr of mentions hasn't changed and that the nr of concepts is not less than
        # before
        assert len(cpt_id_to_mentions) == len(cpt_id_to_cpt_details)
        assert len(cpt_id_to_mentions) >= nr_concepts_old

        cnt_cpt_mens = sum([len(cl) for cl in cpt_id_to_mentions.values()])
        assert len(curr_mentions) == cnt_cpt_mens
        assert nr_mentions_old == len(curr_mentions)

    to_ret_dwie_format['mentions'] = list()
    for curr_mention in curr_mentions:
        to_ret_dwie_format['mentions'].append({
            'begin': curr_mention['begin'],
            'end': curr_mention['end'],
            'text': curr_mention['text'],
            'tags': [curr_mention['type']],
            'concept': curr_mention['concept']
        })

    to_ret_dwie_format['concepts'] = list()

    for curr_concept_id in range(len(cpt_id_to_mentions)):
        curr_concept_details = cpt_id_to_cpt_details[curr_concept_id]

        assert curr_concept_id == curr_concept_details['id']
        # print('curr concept details: ', curr_concept_details)
        assert len(curr_concept_details['link']) <= 1
        curr_link = None
        if len(curr_concept_details['link']) > 0:
            curr_link = list(curr_concept_details['link'])[0]
        to_ret_dwie_format['concepts'].append({
            'concept': curr_concept_id,
            'text': curr_concept_details['text'],
            'count': len(cpt_id_to_mentions[curr_concept_id]),
            'link': curr_link,
            'tags': list(curr_concept_details['type'])
        })
        to_ret_dwie_format['relations'] = []
    return to_ret_dwie_format


def assert_segment_link_ner(curr_mention, curr_conll2003_segments, curr_aida_segments):
    # print('asserting curr_mention: ', curr_mention, ' vs ', curr_aida_segments)
    assert curr_mention['begin'] == int(curr_conll2003_segments[0])
    assert curr_mention['end'] == int(curr_conll2003_segments[1])
    assert curr_mention['type'] == curr_conll2003_segments[2]
    assert curr_mention['text'].replace('\n', '\\n').replace('\t', '\\t') == curr_conll2003_segments[3]

    assert curr_mention['begin'] == int(curr_aida_segments[0])
    assert curr_mention['end'] == int(curr_aida_segments[1])

    assert curr_mention['text'].replace('\n', '\\n').replace('\t', '\\t') == curr_aida_segments[3]

    if curr_mention['link'] is not None:
        assert curr_mention['link'] == curr_aida_segments[4]
    else:
        assert curr_aida_segments[2] == 'nill'


if __name__ == "__main__":
    conll2003_ner_types = {'LOC', 'MISC', 'ORG', 'PER'}
    base_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida'

    # processing conf to also export my coref annotations
    # processing_conf = [
    #     {
    #         'tag': 'testa',
    #         'conll2003_orig': 'conll2003/eng.testa',
    #         'conll2003_johannes': 'johannes/eng.testa.txt',
    #         'conll2003_segments': 'johannes/conll-testa.segments',
    #         'conll2003_corpus': 'johannes/testa.docs',
    #         'aida_orig': 'aida/partitioned/aida_testa.tsv',
    #         'aida_segments': 'johannes/testa.segments',
    #         'aida_coref_annos_klim': 'aida/partitioned/aida_testa.tsv.coref.tsv',
    #         'output_path_file': 'aida/partitioned/aida_dwie_format/testa.jsonl',
    #         'output_path_dir': 'aida/partitioned/aida_dwie_format/testa/'
    #     },
    #     {
    #         'tag': 'train',
    #         'conll2003_orig': 'conll2003/eng.train',
    #         'conll2003_johannes': 'johannes/eng.train.txt',
    #         'conll2003_segments': 'johannes/conll-train.segments',
    #         'conll2003_corpus': 'johannes/train.docs',
    #         'aida_orig': 'aida/partitioned/aida_train.tsv',
    #         'aida_segments': 'johannes/train.segments',
    #         'aida_coref_annos_klim': 'aida/partitioned/aida_train.tsv.coref.tsv',
    #         'output_path_file': 'aida/partitioned/aida_dwie_format/train.jsonl',
    #         'output_path_dir': 'aida/partitioned/aida_dwie_format/train/'
    #     },
    #     {
    #         'tag': 'testb',
    #         'conll2003_orig': 'conll2003/eng.testb',
    #         'conll2003_johannes': 'johannes/eng.testb.txt',
    #         'conll2003_segments': 'johannes/conll-testb.segments',
    #         'conll2003_corpus': 'johannes/testb.docs',
    #         'aida_orig': 'aida/partitioned/aida_testb.tsv',
    #         'aida_segments': 'johannes/testb.segments',
    #         'aida_coref_annos_klim': 'aida/partitioned/aida_testb.tsv.coref.tsv',
    #         'output_path_file': 'aida/partitioned/aida_dwie_format/testb.jsonl',
    #         'output_path_dir': 'aida/partitioned/aida_dwie_format/testb/'
    #     }
    # ]

    # processing conf without exporting my coref annotations
    processing_conf = [
        {
            'tag': 'testa',
            'conll2003_orig': 'conll2003/eng.testa',
            'conll2003_johannes': 'johannes/eng.testa.txt',
            'conll2003_segments': 'johannes/conll-testa.segments',
            'conll2003_corpus': 'johannes/testa.docs',
            'aida_orig': 'aida/partitioned/aida_testa.tsv',
            'aida_segments': 'johannes/testa.segments',
            'aida_coref_annos_klim': 'aida/partitioned/nonexistent.tsv',
            'output_path_file': 'aida/partitioned/aida_dwie_format_no_coref_annos/testa.jsonl',
            'output_path_dir': 'aida/partitioned/aida_dwie_format_no_coref_annos/testa/'
        },
        {
            'tag': 'train',
            'conll2003_orig': 'conll2003/eng.train',
            'conll2003_johannes': 'johannes/eng.train.txt',
            'conll2003_segments': 'johannes/conll-train.segments',
            'conll2003_corpus': 'johannes/train.docs',
            'aida_orig': 'aida/partitioned/aida_train.tsv',
            'aida_segments': 'johannes/train.segments',
            'aida_coref_annos_klim': 'aida/partitioned/nonexistent.tsv',
            'output_path_file': 'aida/partitioned/aida_dwie_format_no_coref_annos/train.jsonl',
            'output_path_dir': 'aida/partitioned/aida_dwie_format_no_coref_annos/train/'
        },
        {
            'tag': 'testb',
            'conll2003_orig': 'conll2003/eng.testb',
            'conll2003_johannes': 'johannes/eng.testb.txt',
            'conll2003_segments': 'johannes/conll-testb.segments',
            'conll2003_corpus': 'johannes/testb.docs',
            'aida_orig': 'aida/partitioned/aida_testb.tsv',
            'aida_segments': 'johannes/testb.segments',
            'aida_coref_annos_klim': 'aida/partitioned/nonexistent.tsv',
            'output_path_file': 'aida/partitioned/aida_dwie_format_no_coref_annos/testb.jsonl',
            'output_path_dir': 'aida/partitioned/aida_dwie_format_no_coref_annos/testb/'
        }
    ]

    print('step 1 - conll2003 corpus extractor')
    nr_docs_read = 0
    tot_nr_docs_read = 0
    for curr_conf in processing_conf:
        dwie_format_output = list()
        file = open(os.path.join(base_path, curr_conf['conll2003_corpus']), 'rb')
        nr_docs_read = 0
        print('=======processing {}======='.format(curr_conf['tag']))

        # lst_conll2003_orig = load_in_list(os.path.join(base_path, curr_conf['conll2003_orig']), ' ')
        # lst_conll2003_johannes = load_in_list(os.path.join(base_path, curr_conf['conll2003_johannes']), '\t')
        # lst_conll2003_segments = load_in_list(os.path.join(base_path, curr_conf['conll2003_segments']), '\t')
        # lst_aida_orig = load_in_list(os.path.join(base_path, curr_conf['aida_orig']), '\t')
        # lst_aida_segments = load_in_list(os.path.join(base_path, curr_conf['aida_segments']), '\t')
        # lst_aida_coref_annos = load_in_list(os.path.join(base_path, curr_conf['aida_coref_annos_klim']), '\t')
        file_conll2003_orig = open(os.path.join(base_path, curr_conf['conll2003_orig']))
        file_conll2003_johannes = open(os.path.join(base_path, curr_conf['conll2003_johannes']))
        file_conll2003_segments = open(os.path.join(base_path, curr_conf['conll2003_segments']))
        file_aida_orig = open(os.path.join(base_path, curr_conf['aida_orig']))
        file_aida_segments = open(os.path.join(base_path, curr_conf['aida_segments']))
        file_aida_coref_annos = None
        if os.path.isfile(os.path.join(base_path, curr_conf['aida_coref_annos_klim'])):
            file_aida_coref_annos = open(os.path.join(base_path, curr_conf['aida_coref_annos_klim']))

        while True:
            curr_sentence = -1
            tot_nr_docs_read += 1
            nr_docs_read += 1
            ptr_raw_content = 0
            doc_raw_content: str
            doc_id, doc_raw_content = get_next_article(file)
            if doc_id is None:
                break
            doc_started = False
            nr_doc_tokens = 0
            # curr_mention = {'begin': None, 'end': None, 'concept': None, 'type': None, 'text': None, 'link': None}
            curr_mention = None
            curr_doc_mentions = list()
            # curr_concept_details = {'id': None, 'text': None, 'type': {}, 'link': {}}
            link_to_concept_id = dict()
            curr_concept_id = None
            concept_id_to_mentions = dict()
            concept_id_to_concept_details = dict()
            curr_doc_tokenization = {'begin': list(), 'end': list(), 'tokens': list(), 'sentences': list()}

            prev_other = True
            # prev_mention = False
            added_concept_ids = set()
            # for ptr_conll2003_orig, content in enumerate(lst_conll2003_orig[ptr_conll2003_orig:], ptr_conll2003_orig):
            for content in file_conll2003_orig:
                content = parse_tsv(content, ' ')
                # content = lst_conll2003_orig.pop(0)
                if content[0] == '':
                    curr_sentence += 1
                    continue

                curr_conll2003_johannes = get_next_ne_line_file(file_conll2003_johannes, '\t')

                curr_aida_orig = get_next_ne_line_file(file_aida_orig, '\t')

                if file_aida_coref_annos is not None:
                    curr_aida_coref_annos = get_next_ne_line_file(file_aida_coref_annos, '\t')
                else:
                    curr_aida_coref_annos = None

                if content[0] == '-DOCSTART-':
                    curr_sentence = -1
                    curr_doc_info = curr_conll2003_johannes[0].split(' ')
                    assert '#doc' in curr_doc_info[0]
                    curr_conll2003_segments = get_next_ne_line_file(file_conll2003_segments, '\t')
                    curr_aida_segments = get_next_ne_line_file(file_aida_segments, '\t')
                    assert '#doc' in curr_aida_segments[0] or curr_mention is not None
                    assert '#doc' in curr_conll2003_segments[0] or curr_mention is not None
                    if curr_mention is not None:
                        # assert with Johannes segment link, ner types, and text
                        assert_segment_link_ner(curr_mention, curr_conll2003_segments, curr_aida_segments)
                        # assert curr_mention['begin'] == int(curr_conll2003_segments[0])
                        # assert curr_mention['end'] == int(curr_conll2003_segments[1])
                        # assert curr_mention['type'] == curr_conll2003_segments[2]
                        # assert curr_mention['text'] == curr_conll2003_segments[3]

                        curr_conll2003_segments = get_next_ne_line_file(file_conll2003_segments, '\t')
                        curr_aida_segments = get_next_ne_line_file(file_aida_segments, '\t')
                        assert '#doc' in curr_aida_segments[0]
                        assert '#doc' in curr_conll2003_segments[0]
                        # end assert with Johannes segment link, ner types, and text

                    assert '-DOCSTART-' in curr_aida_orig[0]
                    if curr_aida_coref_annos is not None:
                        assert '-DOCSTART-' in curr_aida_coref_annos[0]
                    # print('processing the document ', curr_conll2003_johannes)
                    if doc_started:
                        # TODO: save the article, add the corresponding file/line in DWIE format with annotations,
                        #  and break
                        break
                    # assert tot_nr_docs_read == int(curr_doc_info[1])
                else:
                    # assert doc_started
                    assert content[0] == curr_conll2003_johannes[0]
                    assert content[0] == curr_aida_orig[0]
                    if curr_aida_coref_annos is not None:
                        assert content[0] == curr_aida_coref_annos[0]
                    index_token_begin = doc_raw_content.index(content[0], ptr_raw_content)
                    index_token_end = index_token_begin + len(content[0])

                    # checking that there is no meaningful content ommited from the raw
                    assert doc_raw_content[ptr_raw_content:index_token_begin].strip() == ''
                    ptr_raw_content = index_token_end
                    curr_doc_tokenization['begin'].append(index_token_begin)
                    curr_doc_tokenization['end'].append(index_token_end)
                    curr_raw_token = doc_raw_content[index_token_begin: index_token_end]
                    assert curr_raw_token == content[0]
                    curr_doc_tokenization['tokens'].append(curr_raw_token)
                    curr_doc_tokenization['sentences'].append(curr_sentence)

                    # TODO: adds NER type
                    if content[3] != 'O':
                        assert 'I' == curr_aida_orig[1] or 'B' == curr_aida_orig[1]
                        assert (not prev_other) or 'B' == curr_aida_orig[1]

                        if 'B' == curr_aida_orig[1]:
                            if curr_mention is not None:
                                assert curr_concept_id is not None
                                # checks with segments provided by Johannes to be sure we do not miss any entity mention
                                curr_conll2003_segments = get_next_ne_line_file(file_conll2003_segments, '\t')
                                curr_aida_segments = get_next_ne_line_file(file_aida_segments, '\t')
                                assert_segment_link_ner(curr_mention, curr_conll2003_segments, curr_aida_segments)

                                # assert curr_mention['begin'] == int(curr_conll2003_segments[0])
                                # assert curr_mention['end'] == int(curr_conll2003_segments[1])
                                # assert curr_mention['type'] == curr_conll2003_segments[2]
                                # print('ready done')

                                # end checks with segments provided by Johannes

                                curr_doc_mentions.append(curr_mention)
                                concept_id_to_concept_details, concept_id_to_mentions = \
                                    add_curr_mention(curr_mention,
                                                     curr_concept_id,
                                                     concept_id_to_concept_details,
                                                     concept_id_to_mentions)

                            curr_mention = {'begin': index_token_begin,
                                            'end': None,
                                            'concept': None,
                                            'type': None,
                                            'text': None,
                                            'link': None}
                            curr_concept_id = None

                        curr_mention['end'] = index_token_end
                        curr_mention['text'] = doc_raw_content[curr_mention['begin']:curr_mention['end']]

                        assert content[3][2:] in conll2003_ner_types

                        curr_mention['type'] = content[3][2:]
                        if curr_aida_orig[3].upper() != '--NME--':
                            curr_mention['link'] = curr_aida_orig[3]

                        if curr_concept_id is None:
                            if curr_aida_coref_annos is not None:
                                if curr_aida_coref_annos[3] != '--NME--':
                                    # print('curr aida coref annos: ', curr_aida_coref_annos)
                                    curr_concept_id = curr_aida_coref_annos[8]
                                    curr_concept_id = int(curr_concept_id)
                                    # added_concept_ids.add(curr_concept_id)
                                    assert content[3][2:] == curr_aida_coref_annos[7][2:]
                                else:
                                    curr_concept_id = curr_aida_coref_annos[5]
                                    curr_concept_id = int(curr_concept_id)
                                    # added_concept_ids.add(curr_concept_id)
                                    assert content[3][2:] == curr_aida_coref_annos[4][2:]
                            else:
                                if curr_mention['link'] is not None:
                                    if curr_mention['link'] in link_to_concept_id:
                                        curr_concept_id = link_to_concept_id[curr_mention['link']]
                                    else:
                                        curr_concept_id = get_next_id(added_concept_ids)
                                        link_to_concept_id[curr_mention['link']] = curr_concept_id
                                        # curr_concept_id = link_to_concept_id[curr_mention['link']]
                                else:
                                    # curr_concept_id = max(added_concept_ids) + 1
                                    curr_concept_id = get_next_id(added_concept_ids)

                        # added_concept_ids.add(curr_concept_id)
                        curr_mention['concept'] = curr_concept_id
                        prev_other = False
                    else:
                        if not prev_other:
                            assert curr_concept_id is not None

                            # assert with Johannes segment link, ner types, and text
                            curr_conll2003_segments = get_next_ne_line_file(file_conll2003_segments, '\t')
                            curr_aida_segments = get_next_ne_line_file(file_aida_segments, '\t')
                            assert_segment_link_ner(curr_mention, curr_conll2003_segments, curr_aida_segments)

                            # assert curr_mention['begin'] == int(curr_conll2003_segments[0])
                            # assert curr_mention['end'] == int(curr_conll2003_segments[1])
                            # assert curr_mention['type'] == curr_conll2003_segments[2]
                            # end assert with Johannes segment link, ner types, and text

                            curr_doc_mentions.append(curr_mention)
                            concept_id_to_concept_details, concept_id_to_mentions = \
                                add_curr_mention(curr_mention,
                                                 curr_concept_id,
                                                 concept_id_to_concept_details,
                                                 concept_id_to_mentions)

                            prev_other = True
                            curr_concept_id = None
                            curr_mention = None

                    # asserting that begin and end matches with what Johannes has calculated before

                doc_started = True
                nr_doc_tokens += 1

            # no more mentions to process, for now just assert, if it is false (there is a document whose last token
            # belongs to an entity mention), then will have to add the logic to process the pending curr_mention here
            # assert curr_mention is None
            if curr_mention is not None:
                assert curr_concept_id is not None

                curr_doc_mentions.append(curr_mention)
                concept_id_to_concept_details, concept_id_to_mentions = \
                    add_curr_mention(curr_mention,
                                     curr_concept_id,
                                     concept_id_to_concept_details,
                                     concept_id_to_mentions)
                curr_mention = None

            # now adds a new entry in the DWIE format
            curr_dwie_doc = get_in_dwie_format(curr_doc_mentions, concept_id_to_mentions,
                                               concept_id_to_concept_details, curr_doc_tokenization,
                                               curr_conf['tag'], doc_id, doc_raw_content,
                                               break_multiple_ner_types=False)
            dwie_format_output.append(curr_dwie_doc)

        save_dwie_in_jsonl(dwie_format_output, os.path.join(base_path, curr_conf['output_path_file']))
        save_dwie_in_json_files(dwie_format_output, os.path.join(base_path, curr_conf['output_path_dir']))
