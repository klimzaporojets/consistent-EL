import json

if __name__ == "__main__":
    path_file = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/local_tests/' \
                'dwie-klim-debug-1-spanbert-only-nil-clusters-mtt/DW_187058777.json'

    path_file_out = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/local_tests/' \
                    'dwie-klim-debug-1-spanbert-only-nil-clusters-mtt/DW_187058777_out.json'

    with open(path_file, 'rt') as infile:
        parsed_json = json.load(infile)
        new_candidates = []
        new_targets = []
        new_scores = []
        for candidates in parsed_json['all_spans_candidates']:
            new_candidates.append([0])

        for targets in parsed_json['all_spans_candidates_target']:
            new_targets.append(0)

        for targets in parsed_json['all_spans_candidates_scores']:
            new_scores.append([1.0])

        parsed_json['all_spans_candidates'] = new_candidates
        parsed_json['all_spans_candidates_target'] = new_targets
        parsed_json['all_spans_candidates_scores'] = new_scores

    with open(path_file_out, 'wt') as outfile:
        json.dump(parsed_json, outfile)
