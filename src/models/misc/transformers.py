import torch
import torch.nn as nn
from transformers import *

from data_processing.bert_preprocessing import get_segmented_doc_for_bert, BertDocument
from misc import settings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def mysentsplitter(tokens, maxlen):
    sentences = []
    begin = 0
    while begin < len(tokens):
        if len(tokens) - begin < maxlen:
            end = len(tokens)
        else:
            end = begin + maxlen
            while end > begin and tokens[end - 1] != '.':
                end -= 1
            if begin == end:
                logger.warning('FAILED TO SPLIT INTO SENTENCES: %s' % tokens[begin:])
                end = begin + maxlen
        sentences.append(tokens[begin:end])
        begin = end
    return sentences


def myencode(tokenizer, orig_tokens):
    bert_tokens = []
    orig_to_tok_map = []

    bert_tokens.append('[CLS]')
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append('[SEP]')

    return bert_tokens, orig_to_tok_map


def pad_tensors(instances):
    maxlen = max([x.size()[1] for x in instances])
    out = []
    for instance in instances:
        if instance.size()[1] < maxlen:
            instance = torch.cat((instance, torch.zeros(1, maxlen - instance.size()[1], instance.size()[2]).cuda()), 1)
        out.append(instance)
    return torch.cat(out, 0)


class CombineConcat(nn.Module):
    def __init__(self):
        super(CombineConcat, self).__init__()

    def forward(self, list_of_tensors):
        return torch.cat(list_of_tensors, -1)


class WrapperBERT(nn.Module):

    def __init__(self, config):
        super(WrapperBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.layers = config['layers']
        self.max_bert_length = config['max_length']

        if config['combine'] == 'concat':
            self.out = CombineConcat()
            self.dim_output = 768 * len(self.layers)
        else:
            raise BaseException('no such module %s ' % config['combine'])

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        instances = []
        for text in texts:
            reps = [list() for _ in self.layers]

            for sentence in mysentsplitter(text, self.max_bert_length):
                tokens, orig_to_tok_map = myencode(self.tokenizer, sentence)

                enc_toks = self.tokenizer.encode(tokens)

                if settings.device == 'cuda':
                    input_ids = torch.cuda.LongTensor(enc_toks).unsqueeze(0)
                else:
                    input_ids = torch.LongTensor(enc_toks).unsqueeze(0)
                outputs = self.model(input_ids)
                all_hidden_states, all_attentions = outputs[-2:]

                if settings.device == 'cuda':
                    indices = torch.cuda.LongTensor(orig_to_tok_map)
                else:
                    indices = torch.LongTensor(orig_to_tok_map)

                for rep, l in zip(reps, self.layers):
                    rep.append(torch.index_select(all_attentions[l].detach(), 1, indices))

            instances.append([torch.cat(rep, 1) for rep in reps])
        # transpose
        instances = list(map(list, zip(*instances)))
        # pad layers
        instances = [pad_tensors(x).detach() for x in instances]
        output = self.out(instances)
        return output


class WrapperSpanBERT(nn.Module):

    def __init__(self, config):
        super(WrapperSpanBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model_path = config['model_path']
        self.spanbert_model = BertModel.from_pretrained(self.model_path)
        self.max_bert_length = config['max_length']

        if config['combine'] == 'concat':
            self.dim_output = self.spanbert_model.config.hidden_size
        else:
            raise BaseException("no such module")

        for param in self.spanbert_model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        assert len(texts) == 1  # only for batch size 1 for now
        rep = list()
        for text in texts:
            for sentence in mysentsplitter(text, self.max_bert_length):
                tokens, orig_to_tok_map = myencode(self.tokenizer, sentence)

                # (kzaporoj 18/03/2021) - the tokens are already bert-tokenized (["[CLS]", "##..", ...]),
                #  so no need to .encode, it only splits the "[CLS]", which is not needed, this is why the encode is
                #  commented and replaced by convert_tokens_to_ids.
                enc_toks = self.tokenizer.convert_tokens_to_ids(tokens)
                if settings.device == 'cuda':
                    input_ids = torch.cuda.LongTensor(enc_toks).unsqueeze(0)
                else:
                    input_ids = torch.LongTensor(enc_toks).unsqueeze(0)

                outputs = self.spanbert_model(input_ids).last_hidden_state

                if settings.device == 'cuda':
                    indices = torch.cuda.LongTensor(orig_to_tok_map)
                else:
                    indices = torch.LongTensor(orig_to_tok_map)

                rep.append(torch.index_select(outputs, 1, indices))

        return torch.cat(rep, 1)


class WrapperSpanBERTSubtoken(nn.Module):
    """ The goal of this class is to reproduce the SpanBert tokenization and segmentation as in the current coref
    state-of-the-art paper (https://www.aclweb.org/anthology/2020.emnlp-main.686).
    Some of the code is adapted from the github of this paper: https://github.com/lxucs/coref-hoi .
    For example, one of the differences seems that WrapperSpanBERT measures segmentation in words (tokens), while
    the coref-hoi does it in BERT sub-tokens, so the hyperparameter max_bert_length is on BERT sub-token level. """

    def __init__(self, config):
        super(WrapperSpanBERTSubtoken, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model_path = config['model_path']
        self.spanbert_model = BertModel.from_pretrained(self.model_path)
        self.max_bert_length = config['max_length']

        if config['combine'] == 'concat':
            self.dim_output = self.spanbert_model.config.hidden_size
        else:
            raise BaseException("no such module")

        for param in self.spanbert_model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        assert len(texts) == 1  # only for batch size 1 for now
        rep = list()
        for text in texts:
            # for sentence in mysentsplitter(text, self.max_bert_length):
            # def get_segmented_doc_for_bert(text, seg_len, tokenizer):
            segmented_doc_for_bert: BertDocument = get_segmented_doc_for_bert(text, self.max_bert_length,
                                                                              self.tokenizer)
            # for sentence in mysentsplitter(text, self.max_bert_length):
            for sentence in segmented_doc_for_bert.segments:
                tokens, orig_to_tok_map = myencode(self.tokenizer, sentence)

                enc_toks = self.tokenizer.convert_tokens_to_ids(tokens)
                if settings.device == 'cuda':
                    input_ids = torch.cuda.LongTensor(enc_toks).unsqueeze(0)
                else:
                    input_ids = torch.LongTensor(enc_toks).unsqueeze(0)

                outputs = self.spanbert_model(input_ids).last_hidden_state

                if settings.device == 'cuda':
                    indices = torch.cuda.LongTensor(orig_to_tok_map)
                else:
                    indices = torch.LongTensor(orig_to_tok_map)

                rep.append(torch.index_select(outputs, 1, indices))

        return torch.cat(rep, 1)


class WrapperSpanBERT_X(nn.Module):
    """ The goal of this class IS TO USE the SpanBert segmentation (and tokenization). So no tokenization has to be
    done here as is in WrapperSpanBERTSubtoken. The tokenization/segmentation is done by a separate class:
    main_bert_processor.py.  In other words, the input is already segmented and bert-tokenized.
    Haven't come with the right name for this class this is why it ends in _X.
    Some of the ideas/code is taken from https://github.com/lxucs/coref-hoi .
    """

    def __init__(self, config):
        super(WrapperSpanBERT_X, self).__init__()

        self.model_path = config['model_path']
        self.spanbert_model = BertModel.from_pretrained(self.model_path)

        self.dim_output = self.spanbert_model.config.hidden_size

        if 'fine_tune_bert' not in config or config['fine_tune_bert'] is False:
            for param in self.spanbert_model.parameters():
                param.requires_grad = False

    def forward(self, segmented_docs, segment_masks):
        assert len(segmented_docs) == 1  # only for batch size 1 for now
        rep = list()
        for segmented_doc, segments_mask in zip(segmented_docs, segment_masks):
            outputs = self.spanbert_model(segmented_doc, attention_mask=segments_mask)[0]
            segments_mask = segments_mask.to(torch.bool)
            outputs = outputs[segments_mask]
            rep.append(outputs)

        return torch.cat(rep, 1).unsqueeze(0)
