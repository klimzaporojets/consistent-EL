import torch

import settings
from cpn.data_reader_api import DataReaderAPI
from models import MyDygie3
from models.coreflinker.dygie3 import collate_dygie
from traintool import load_model


class NNProcessing:
    def __init__(self, config, args):
        # load_datasets_from_config is false because the server api takes as input any text from the user, it is not bound
        # to a specific dataset
        loaded_model_dict = load_model(config, training=False, load_datasets_from_config=False)
        loaded_model: MyDygie3 = loaded_model_dict['model']
        dictionaries = loaded_model_dict['dictionaries']
        linking_candidates = loaded_model_dict['linking_candidates']

        with open(args.model_path, 'rb') as f:
            loaded_model.load_model(f, to_cpu=True, load_word_embeddings=False)  # for now only on cpu

        loaded_model.eval()  # only eval mode
        self.loaded_model = loaded_model
        self.data_reader_api = DataReaderAPI(config, dictionaries, linking_candidates)

    def format_short(self, output_json):
        """

        :param output_json:
        :return: example: [(0, 4, 'United_States'), (242, 9, 'Sago_Mine_disaster'), (1281, 9, 'Sago_Mine_disaster'),
        (2173, 30, 'United_Mine_Workers'), (62, 9, 'Sago_Mine_disaster'), (371, 13, 'Sago_Mine_disaster')]

        TODO: maybe also add the mention text.
        """
        men_to_ret = []
        for curr_mention in output_json['mentions']:
            curr_concept_id = curr_mention['concept']
            pred_link = output_json['concepts'][curr_concept_id]['link_pred']
            # if pred_link is not None:
            #     men_to_ret.append((curr_mention['begin'], curr_mention['end'] - curr_mention['begin'],
            #                        curr_mention['text'], pred_link))

            # for now adds mention even if the link was none
            men_to_ret.append((curr_mention['begin'], curr_mention['end'] - curr_mention['begin'],
                               curr_mention['text'], pred_link))

        return men_to_ret

    def process(self, input_json):
        # print('getting the result with the following input: ', input_json)
        read_input = self.data_reader_api.convert(input_json)
        # print('the read_input is: ', read_input)
        collated_input = collate_dygie(self.loaded_model, [read_input], torch.device(settings.device), collate_api=True)
        # print('the collated_input is: ', collated_input)
        if 'output_config' not in input_json:
            output_config = {
                "output_content": True,
                "_output_content": "Whether the 'content' is added to prediction json file.",
                "output_tokens": True,
                "_output_tokens": "Whether the 'tokens' are added to prediction json file. "
            }
        else:
            output_config = input_json['output_config']
        _, predictions = self.loaded_model.predict(collated_input['inputs'], collated_input['relations'],
                                                   collated_input['metadata'], output_config=output_config)

        predictions = predictions[0]
        if 'format' in input_json:
            json_format = input_json['format']
            if json_format == 'short':
                return self.format_short(predictions)
        return predictions
