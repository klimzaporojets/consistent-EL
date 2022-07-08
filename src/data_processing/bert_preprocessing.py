"""
Various tokenization, segmentation, and other routines to get the SpanBert input tokens.
Most of the logic is based on the current coref state-of-the-art https://github.com/lxucs/coref-hoi/ .
"""
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


def flatten(l):
    return [item for sublist in l for item in sublist]


class BertDocument(object):
    def __init__(self):
        # self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word

    def finalize(self):
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        self.subtoken_map = subtoken_map
        self.sentence_map = sentence_map


def split_into_segments(document_state: BertDocument, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info('no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.info('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


# def get_segmented_doc_for_bert(doc_key, text, language, seg_len, tokenizer):
def get_segmented_doc_for_bert(text, seg_len, tokenizer):
    """ Process raw input to finalized documents """
    segmented_doc_for_bert = BertDocument()
    word_idx = -1

    # Build up documents
    for word in text:
        if word == '.':
            segmented_doc_for_bert.sentence_end[-1] = True
        else:
            word_idx += 1
            word = normalize_word(word)
            subtokens = tokenizer.tokenize(word)
            segmented_doc_for_bert.tokens.append(word)
            segmented_doc_for_bert.token_end += [False] * (len(subtokens) - 1) + [True]
            for idx, subtoken in enumerate(subtokens):
                segmented_doc_for_bert.subtokens.append(subtoken)
                info = None if idx != 0 else len(subtokens)
                segmented_doc_for_bert.info.append(info)
                segmented_doc_for_bert.sentence_end.append(False)
                segmented_doc_for_bert.subtoken_map.append(word_idx)

    # Split documents
    constraits1 = segmented_doc_for_bert.sentence_end
    split_into_segments(segmented_doc_for_bert, seg_len, constraits1, segmented_doc_for_bert.token_end, tokenizer)
    segmented_doc_for_bert.finalize()
    return segmented_doc_for_bert
