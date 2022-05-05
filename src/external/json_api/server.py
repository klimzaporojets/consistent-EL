import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

#
# import model.config as config
# from gerbil.build_entity_universe import BuildEntityUniverse
# from gerbil.nn_processing import NNProcessing
# from model.util import load_train_args
import settings
from external.json_api.nn_processing import NNProcessing
from util.debug import Tee


class GetHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        input_json = read_json(post_data)
        print('processing the following input_json: ', input_json)
        response = nnprocessing.process(input_json)
        print("response in server.py code:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    return data


def _parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_directory", default="models/20201023-coreflinker_e2e-ap0-1",
    parser.add_argument("--experiment_directory", default="config/local_tests/baseline_linking_test/",
                        help="the directory of where the experiment is located (config, models, test.jsonl, etc.)")
    parser.add_argument("--config_file", default="config_server.json",
                        help="the config file, can be also config.json, but that one can have filters for mention"
                             "/entity embeddings which can make the embedding space quite limited "
                             "(limited to dictionary in DWIE dataset). On the other hand, config_server.json"
                             " should have all the 'use_filtered' set in false, among other things "
                             "inherent to e2e server that are not clear right now and will be discovered in future...")
    parser.add_argument("--port", default=5556, type=int, help="the port on which the service will be loaded")

    args = parser.parse_args()

    args.training_config_path = os.path.join(args.experiment_directory, args.config_file)
    args.model_path = os.path.join(args.experiment_directory, 'last.model')

    return args


def terminate():
    tee.close()


if __name__ == "__main__":
    # args, train_args = _parse_args()
    args = _parse_args()
    settings.device = 'cpu'
    with open(args.training_config_path) as f:
        config = json.load(f)

    if 'path' not in config:
        print('api server setting path=', Path(args.training_config_path).parent)
        config['path'] = Path(args.training_config_path).parent

    nnprocessing = NNProcessing(config, args)

    ####################

    server = HTTPServer(('0.0.0.0', args.port), GetHandler)
    print('Starting server at http://localhost:{}'.format(args.port))

    tee = Tee('server.txt', 'w')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)
