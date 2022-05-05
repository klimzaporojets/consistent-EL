# just to be sure that no empty tags are present in concepts for predicted joint file
import json


def is_associated_relation(concept_id, relations):
    for curr_relation in relations:
        if curr_relation['s'] == concept_id or curr_relation['o'] == concept_id:
            return True
    return False


def is_associated_tags_mentions(concept_id, mentions):
    allzero = True
    for curr_mention in mentions:
        if curr_mention['concept'] == concept_id and len(curr_mention['tags']) > 0:
            allzero = False
    return allzero


if __name__ == "__main__":
    path_test_file = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/models/' \
                     '20200620_joint_rp0_ap0_cp0-1/test.json'
    with open(path_test_file) as infile:
        for curr_line in infile:
            parsed_file = json.loads(curr_line)
            for curr_concept in parsed_file['concepts']:
                if len(curr_concept['tags']) == 0:
                    associate_relation = is_associated_relation(curr_concept['concept'], parsed_file['relations'])
                    mentions_tags = is_associated_tags_mentions(curr_concept['concept'], parsed_file['mentions'])
                    print('LEN IN 0 mentions in concept: ', curr_concept['count'], ' file: ', parsed_file['id'],
                          'concept id: ', curr_concept['concept'], ' part of relation: ', associate_relation,
                          'mentions no tags: ', mentions_tags)
