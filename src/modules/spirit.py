import torch
import torch.nn as nn
# from allennlp.common.util import pad_sequence_to_length
# from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
# from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
# from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
# from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder

from modules.seq2seq import seq2seq_create
from modules.text_field import TextFieldEmbedderTokens, TextFieldEmbedderCharacters, TextFieldEmbedderWhitespace
from modules.transformers import WrapperBERT, WrapperSpanBERT, WrapperSpanBERTSubtoken, WrapperSpanBERT_X
from util.debug import Wrapper1
from util.nn import create_activation_function


class WrapperElmoTokenEmbedder(nn.Module):

    # def __init__(self, dictionaries, config):
        # super(WrapperElmoTokenEmbedder, self).__init__()
        # self.mapper = ELMoCharacterMapper()
        # self.elmo = ElmoTokenEmbedder(
        #     config['options_file'],
        #     config['weight_file'],
        #     config['do_layer_norm'],
        #     config['dropout']
        # )
        # self.dim = 1024
        # pass

    @staticmethod
    def _default_value_for_padding():
        return [0] * 50

    def forward(self, texts):
        pass
        # out = [[self.mapper.convert_word_to_char_ids(w) for w in text] for text in texts]
        # max_len = max([len(x) for x in out])
        # out = [pad_sequence_to_length(x, max_len, default_value=self._default_value_for_padding) for x in out]
        # out = torch.LongTensor(out).cuda()
        # return self.elmo(out)


# class EntityEmbedder(nn.Module):
#     def __init__(self, dictionaries, config):
#         super(EntityEmbedder, self).__init__()
#         self.config = config
#         self.dim_output = 0
#         if 'yamada_embedder' in config:
#             self.entity_embedder_yamada = TextFieldEmbedderTokens(dictionaries, config['yamada_embedder'])
#             self.dim_output += self.entity_embedder_yamada.dim
#
#     def forward(self, data):
#         outputs = []
#         if 'yamada_embedder' in self.config:
#             outputs.append(self.entity_embedder_yamada(data['tokens'], data['entities']))
#
#         return torch.cat(outputs, -1)


class TextEmbedder(nn.Module):
    def __init__(self, dictionaries, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.dim_output = 0
        if 'char_embedder' in config:
            self.char_embedder = TextFieldEmbedderCharacters(dictionaries, config['char_embedder'])
            self.dim_output += self.char_embedder.dim_output
            self.do_char_embedding = True
        else:
            self.do_char_embedding = False
        if 'text_field_embedder' in config:
            self.word_embedder = TextFieldEmbedderTokens(dictionaries, config['text_field_embedder'])
            self.dim_output += self.word_embedder.dim
        if 'whitespace_embedder' in config:
            self.whitespace_embedder = TextFieldEmbedderWhitespace(dictionaries, config['whitespace_embedder'])
            self.dim_output += self.whitespace_embedder.dim
        if 'elmo_embedder' in config:
            self.ctxt_embedder = WrapperElmoTokenEmbedder(dictionaries, config['elmo_embedder'])
            self.dim_output += self.ctxt_embedder.dim
        if 'bert_embedder' in config:
            self.bert_embedder = WrapperBERT(dictionaries, config['bert_embedder'])
            self.dim_output += self.bert_embedder.dim_output
        if 'spanbert_embedder' in config:
            self.spanbert_embedder = WrapperSpanBERT(dictionaries, config['spanbert_embedder'])
            self.dim_output += self.spanbert_embedder.dim_output
        if 'spanbert_embedder_subtoken' in config:
            self.spanbert_embedder = WrapperSpanBERTSubtoken(dictionaries, config['spanbert_embedder_subtoken'])
            self.dim_output += self.spanbert_embedder.dim_output

        if 'spanbert_embedder_x' in config:
            self.spanbert_embedder = WrapperSpanBERT_X(dictionaries, config['spanbert_embedder_x'])
            self.dim_output += self.spanbert_embedder.dim_output

        # if 'entity_embedder' in config:
        #     self.entity_embedder = EntityEmbedder(dictionaries, config['entity_embedder'])
        #     self.dim_output += self.entity_embedder.dim_output

    def forward(self, data):
        outputs = []
        if 'char_embedder' in self.config:
            outputs.append(self.char_embedder(data['characters']))
        if 'text_field_embedder' in self.config:
            outputs.append(self.word_embedder(data['tokens']))
        if 'whitespace_embedder' in self.config:
            outputs.append(self.whitespace_embedder(data))
        if 'elmo_embedder' in self.config:
            outputs.append(self.ctxt_embedder(data['text']))
        if 'bert_embedder' in self.config:
            outputs.append(self.bert_embedder(data['text']))
            # outputs.append(self.bert_embedder(data['tokens']))
        if 'spanbert_embedder' in self.config:
            outputs.append(self.spanbert_embedder(data['text']))
        if 'spanbert_embedder_subtoken' in self.config:
            outputs.append(self.spanbert_embedder(data['text']))

        if 'spanbert_embedder_x' in self.config:
            outputs.append(self.spanbert_embedder(data['bert_segments'], data['bert_segments_mask']))

        # for x in outputs:
        #     inspect('x', x[0,:,:])

        # for x in outputs:
        #     x_norm = x.norm(dim=-1)
        #     print(x.size(), x_norm.size(), x_norm.min().item(), x_norm.max().max().item())

        return torch.cat(outputs, -1)


class SpanExtractor(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanExtractor, self).__init__()
        self.span_extractor1 = AverageSpanExtractor(dim_input) if config['avg'] else None
        self.span_extractor2 = None
        self.span_extractor3 = None
        # self.span_extractor2 = EndpointSpanExtractor(dim_input) if config['endpoint'] else None
        # self.span_extractor3 = SelfAttentiveSpanExtractor(dim_input) if config['self-attentive'] else None
        self.dim_output = self.get_output_dims()

    def forward(self, inputs, token2mention, span_indices):
        mentions = []
        if self.span_extractor1 is not None:
            mentions.append(self.span_extractor1(inputs, token2mention))
        if self.span_extractor2 is not None:
            mentions.append(self.span_extractor2(inputs, span_indices)),
        if self.span_extractor3 is not None:
            mentions.append(self.span_extractor3(inputs, span_indices))
        return torch.cat(mentions, -1)

    def get_output_dims(self):
        dims = 0
        if self.span_extractor1 is not None:
            dims += self.span_extractor1.dim_output
        if self.span_extractor2 is not None:
            dims += self.span_extractor2.get_output_dim()
        if self.span_extractor3 is not None:
            dims += self.span_extractor3.get_output_dim()
        return dims


class AverageSpanExtractor(nn.Module):

    def __init__(self, dim_input):
        super(AverageSpanExtractor, self).__init__()
        self.dim_output = dim_input

    def forward(self, sequence_tensor, span_matrix):
        num_batch = sequence_tensor.size()[0]
        y = sequence_tensor.view(-1, self.dim_output)
        spans = torch.matmul(span_matrix, y)
        spans = spans.view(num_batch, -1, self.dim_output)
        return spans


class ResLayerX(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayerX, self).__init__()
        self.layer = Wrapper1('res', FeedForward(dim_input, config['layer']))
        self.out = nn.Linear(self.layer.dim_output, dim_input)

    def forward(self, tensor):
        return tensor + self.out(self.layer(tensor))


class ResLayer(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayer, self).__init__()
        self.dp = nn.Dropout(config['dropout'])
        self.input = nn.Linear(dim_input, config['dim'])
        self.fnc = create_activation_function(config['actfnc'])
        self.output = nn.Linear(config['dim'], dim_input)

    def forward(self, tensor):
        h = self.dp(tensor)
        h = self.input(h)
        h = self.fnc(h)
        h = self.output(h)
        return tensor + h


class FeedForward(nn.Module):

    def __init__(self, dim_input, config):
        super(FeedForward, self).__init__()
        self.dim_output = dim_input
        self.layers = []

        if 'type' not in config:
            self.create_default(config)
        elif config['type'] == 'ffnn':
            self.create_ffnn(config)
        elif config['type'] == 'res':
            self.create_res(config)
        elif config['type'] == 'resnet':
            self.create_resnet(config)
        elif config['type'] == 'glu':
            self.create_glu(config)
        else:
            raise BaseException("no such type: ", config['type'])

        self.layers = nn.Sequential(*self.layers)

    def create_default(self, config):
        if config['ln']:
            from modules.misc import LayerNorm
            self.layers.append(LayerNorm(self.dim_output))
        if config['dropout'] != 0.0:
            self.layers.append(nn.Dropout(config["dropout"]))

    def create_ffnn(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, dim))
            if 'actfnc' in config:
                self.layers.append(create_activation_function(config['actfnc']))
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def create_glu(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, 2 * dim))
            self.layers.append(nn.GLU())
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def create_res(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayerX(self.dim_output, config))

    def create_resnet(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayer(self.dim_output, config))

    def forward(self, tensor):
        return self.layers(tensor)


class Seq2Seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2Seq, self).__init__()
        self.module = seq2seq_create(dim_input, config)
        self.dim_output = self.module.dim_output

    def forward(self, inputs, seqlens, indices=None):
        return self.module(inputs, seqlens, indices)


class MyBilinear(nn.Module):

    def __init__(self, dim_input1, dim_input2, dim_output):
        super(MyBilinear, self).__init__()
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        self.weight = nn.Bilinear(dim_input1, dim_input2, dim_output)

    def forward(self, inputs1, inputs2):
        batch, length, _ = inputs1.size()
        inputs1 = inputs1.unsqueeze(-2).expand(batch, length, length, self.dim_input1)
        inputs2 = inputs2.unsqueeze(-3).expand(batch, length, length, self.dim_input2)
        out = self.weight(inputs1.contiguous(), inputs2.contiguous())
        return out
