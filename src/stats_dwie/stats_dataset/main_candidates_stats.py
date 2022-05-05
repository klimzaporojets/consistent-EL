# just see how much ram does it take to load all of it
import json

if __name__ == "__main__":
    path_json = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data/' \
                '30092020-end-to-end-johannes/cpn-alias-table.json'

    span_to_candidates = dict()
    for idx, read_line in enumerate(open(path_json)):
        if idx % 10000 == 0:
            print('nr of loaded spans: ', idx)
        curr_load = json.loads(read_line)
        # print('the curr_load is: ', curr_load)
        span_to_candidates[curr_load['text']] = {'candidates': curr_load['candidates'], 'scores': curr_load['scores']}
        # loaded.append(json.loads(read_line))

    print('loaded')
