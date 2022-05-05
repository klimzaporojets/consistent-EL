# The toal of this script is to do two things:
#  1- plot the graph of the size of the adjacency matrix (y-axis) in terms of the % of documents in DWIE (x-axis),
#     to be able to answer the question like: what is the maximum matrix size for < 90% of the documents?
#  2- calculate the determinants of ground truth coreferences and see 1- the time needed, and 2- whether they are
#     mathematically stable.
import json
import os


def plot_coverage_size():
    pass


if __name__ == "__main__":
    dataset_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_public/data/' \
                   'annos_with_content'
    top_linking_candidates = 16
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for filename in filenames:
            # (span start, end) to link, or to id?, NONE if no correct link in candidates
            mention_to_correct_link = dict()
            link_to_id = dict()  # link to id (column) in matrix
            mention_to_concept = dict()  # (begin, end) --> concept id
            mention_to_cluster = dict()  # (begin, end) --> [(begin, end),(begin,end),... all spans in cluster]

            if 'DW_' in filename and 'json' in filename:
                loaded_json = json.load(os.path.join(dirpath, filename))

        pass
