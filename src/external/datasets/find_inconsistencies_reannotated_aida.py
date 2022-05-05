# the main goal of this module is to find inconsistencies in the newly annotated aida (coreference + linking).
# one example of inconsistency (the only one for now) is when a particular mention in a document is not assigned a link,
# BUT the same surface form (the same mention text) has a link in other document. This has to be fixed using
# Johannes' interface:
# The input to this module is the first version of aida in DWIE format exported from the annotations made using
# Johannes' interface (see link above).
import json
import os


def get_url(article_id):
    ids_to_train = set(range(1, 947))
    # blank_url = 'http://10.10.3.23:8083/browser/overlord/051_aida/{}/viewer/viewer?start={}&monitor=&library=&' \
    #             'tag=&q=&view=content.links'

    ids_to_testa = set(range(947, 947 + 216))
    # blank_url = 'http://10.10.3.23:8083/browser/overlord/051_aida/{}/viewer/viewer?start={}&monitor=&library=&' \
    #             'tag=&q=&view=content.links'

    # ids_to_testb = set(range(947 + 216, 947 + 216 + 231))

    # set_name = ''
    # start = article_id
    if article_id in ids_to_train:
        set_name = 'train'
        start = article_id - 1
    elif article_id in ids_to_testa:
        set_name = 'testa'
        start = article_id - 1 - 946
    else:
        set_name = 'testb'
        start = article_id - 1 - 946 - 216

    to_ret_url = 'http://10.10.3.23:8083/browser/overlord/051_aida/{}/viewer/viewer?start={}&monitor=&library=&' \
                 'tag=&q=&view=content.links'.format(set_name, start)

    return to_ret_url


if __name__ == "__main__":
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/aida/' \
                   'aida_reannotated_v1/aida/output-json/current'

    surface_forms_to_links = dict()  # string to a dictionary of associated links (associated with nr each link is used)

    id_to_title = dict()
    # builds link structure
    to_sort_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for curr_filename in filenames:
            if 'main' not in curr_filename:
                to_sort_filenames.append(curr_filename)
                with open(os.path.join(dirpath, curr_filename)) as infile:
                    parsed_json = json.load(infile)
                    article_id = int(parsed_json['id'])
                    for curr_mention in parsed_json['mentions']:
                        curr_concept = parsed_json['concepts'][curr_mention['concept']]
                        if curr_concept['link'] is not None:
                            uni_link = bytes(curr_concept['link'], 'ascii').decode('unicode-escape')
                            if curr_mention['text'] not in surface_forms_to_links:
                                surface_forms_to_links[curr_mention['text']] = dict()

                            if curr_concept['link'] not in surface_forms_to_links[curr_mention['text']]:
                                surface_forms_to_links[curr_mention['text']][uni_link] = dict()
                                surface_forms_to_links[curr_mention['text']][uni_link]['cnt'] = 0
                                surface_forms_to_links[curr_mention['text']][uni_link]['examples'] = set()

                            surface_forms_to_links[curr_mention['text']][uni_link]['cnt'] += 1
                            if len(surface_forms_to_links[curr_mention['text']][uni_link]['examples']) < 10:
                                surface_forms_to_links[curr_mention['text']][uni_link]['examples'] \
                                    .add(get_url(article_id))
            else:
                parsed_main = json.load(open(os.path.join(dirpath, curr_filename)))
                docs = parsed_main['documents']
                for curr_doc in docs:
                    id_to_title[int(curr_doc['id'])] = curr_doc['title']

    to_sort_filenames = sorted(to_sort_filenames, key=lambda x: int(x[0:x.index('.')]))
    curr_nr = 0
    for curr_filename in to_sort_filenames:
        curr_path = os.path.join(dataset_path, curr_filename)
        with open(curr_path) as infile:
            parsed_json = json.load(infile)
            curr_id = int(parsed_json['id'])
            for curr_mention in parsed_json['mentions']:
                curr_concept = parsed_json['concepts'][curr_mention['concept']]
                if curr_concept['link'] is None and curr_mention['text'] in surface_forms_to_links:
                    curr_nr += 1
                    print('{} - =============================================='.format(curr_nr))
                    print('article: ', id_to_title[curr_id])
                    print('Generally the mention "', curr_mention['text'], '" appears is associated to the '
                                                                           'following links: ')
                    print(surface_forms_to_links[curr_mention['text']])
                    print('link to change: ', get_url(curr_id))
                    print('==============================================')

    #     for curr_filename in filenames:
    #         if 'main' not in curr_filename:
    #             with open(os.path.join(dirpath, curr_filename)) as infile:
    #                 parsed_json = json.load(infile)
    #                 curr_id = int(parsed_json['id'])
    #                 for curr_mention in parsed_json['mentions']:
    #                     curr_concept = parsed_json['concepts'][curr_mention['concept']]
    #                     if curr_concept['link'] is None and curr_mention['text'] in surface_forms_to_links:
    #                         curr_nr += 1
    #                         print('{} - =============================================='.format(curr_nr))
    #                         print('article: ', id_to_title[curr_id])
    #                         print('Generally the mention "', curr_mention['text'], '" appears is associated to the '
    #                                                                                'following links: ')
    #                         print(surface_forms_to_links[curr_mention['text']])
    #                         print('link to change: ', get_url(curr_id))
    #                         print('==============================================')
