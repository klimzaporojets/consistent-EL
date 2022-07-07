import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def convert_to_json(identifier, tags, content, begin, end, ner, coref,
                    coref_pointers, coref_scores,
                    rels, mention_rels, links_scores, links_gold,
                    links_pred, singletons=False, output_config=None):
    builder = BuilderDoc(tags, content, begin, end, coref_pointers=coref_pointers, output_config=output_config)
    builder.set_id(identifier)
    builder.singletons = singletons

    for begin, end, tag in ner:
        mention = builder.add_mention(begin, end - 1)  # exclusive
        mention.add_tag(tag)

    if coref is not None:
        for cluster in coref:
            concept = builder.add_concept()
            for begin, end in cluster:
                concept.add_mention(begin, end)

    if rels is not None:
        for src_cluster, dst_cluster, rel in rels:
            src = builder.get_concept(src_cluster)
            dst = builder.get_concept(dst_cluster)
            builder.add_relation(src, dst, rel)

    if mention_rels is not None:
        for src_mention, rel, dst_mention in mention_rels:
            src = builder.add_mention(src_mention[0], src_mention[1])
            dst = builder.add_mention(dst_mention[0], dst_mention[1])
            builder.add_mention_relation(src, dst, rel)

    if links_scores is not None:
        for (begin, end), candidates, scores in links_scores:
            mention = builder.add_mention(begin, end)  # inclusive
            mention.add_candidates_and_scores(candidates, scores)

    if coref_scores is not None:
        # print('in builder the coref_scores are: ', coref_scores)
        for (begin, end), candidates_and_scores in coref_scores.items():
            if (begin, end) in builder.span2mention:
                mention = builder.add_mention(begin, end)
                cand_scores_char_text = [{'span': (builder.begin[k['span'][0]], builder.end[k['span'][1]]),
                                          'score': k['score'],
                                          'text': builder.content[
                                                  builder.begin[k['span'][0]]: builder.end[k['span'][1]]]}
                                         for k in candidates_and_scores]
                mention.add_coref_scores(cand_scores_char_text)

    if links_pred is not None:
        # span_to_pred_link = dict()
        for span_start, span_end, link_pred in links_pred:
            # if not output_config['output_none_mentions']
            mention = builder.add_mention(span_start, span_end)
            # span_to_pred_link[(span_start, span_end)] = link_pred
            mention.add_link_pred(link_pred)

    if links_gold is not None:
        for begin, end, link in links_gold:
            # big fix 03/12/2020 only makes sense to write gold if the mention was already added (ex: through
            # links_pred), if not we will be just adding gold mentions when they are not predicted
            # if (begin, end) in span_to_pred_link:
            #     mention.add_link_pred(span_to_pred_link[(begin, end)])
            if builder.is_existent_mention(begin, end):
                mention = builder.add_mention(begin, end)  # inclusive
                mention.add_link_gold(link)

    return builder.json()


class BuilderDoc:

    def __init__(self, tags, content, begin, end, coref_pointers={}, output_config=None):
        self.tags = tags
        self.content = content
        self.begin = begin
        self.end = end
        self.coref_pointers = coref_pointers
        self.identifier = None
        self.mentions = []
        self.concepts = []
        self.relations = []
        self.mention_relations = []
        self.span2mention = {}
        self.output_config = output_config

    def set_id(self, identifier):
        self.identifier = identifier

    def add_concept(self):
        concept = BuilderConcept(self)
        self.concepts.append(concept)
        return concept

    def is_existent_mention(self, begin, end):
        span = (begin, end)
        return span in self.span2mention

    def add_mention(self, begin, end):
        span = (begin, end)
        if span not in self.span2mention:
            if self.coref_pointers is not None:
                mention = BuilderMention(self, begin, end, self.coref_pointers.get((begin, end)))
            else:
                mention = BuilderMention(self, begin, end, None)

            self.mentions.append(mention)
            self.span2mention[span] = mention
        return self.span2mention[span]

    def add_relation(self, src, dst, rel):
        relation = BuilderRelation(src, dst, rel)
        self.relations.append(relation)
        src.add_relation(relation)
        dst.add_relation(relation)

    def add_mention_relation(self, src, dst, rel):
        relation = BuilderRelation(src, dst, rel)
        self.mention_relations.append(relation)
        src.add_mention_relation(relation)
        dst.add_mention_relation(relation)

    def get_mention(self, begin, end):
        for m in self.mentions:
            if m.token_begin == begin and m.token_end == end:
                return m
        return None

    def get_concept(self, cluster):
        concepts = []
        for begin, end in cluster:
            mention = self.get_mention(begin, end)
            concepts.append(mention.concept)
        for c in concepts:
            if c != concepts[0]:
                logger.info('RELATION HAS MULTIPLE CONCEPTS IN CLUSTER')
        return concepts[0]

    def json(self):
        # sorts mentions by position
        self.mentions = sorted(self.mentions, key=lambda x: x.char_begin)

        # create concepts for mentions without one
        for mention in self.mentions:
            if mention.concept is None:
                self.add_concept().add_mention2(mention)

        # number concepts
        idx = 0
        for concept in self.concepts:
            concept._visible = concept.is_visible()
            if concept._visible:
                concept.idx = idx
                idx += 1

        # number mentions
        idx = 0
        for mention in self.mentions:
            mention._visible = mention.concept._visible or len(mention.mention_relations) > 0 or \
                               mention.candidates is not None or mention.link_gold is not None or \
                               mention.link_pred is not None
            if mention._visible:
                mention.idx = idx
                idx += 1

        tokenization = {}

        if self.output_config['output_tokens']:
            tokenization = {
                'tokens': [self.content[b:e] for b, e in zip(self.begin, self.end)],
                'begin': self.begin,
                'end': self.end}
        return {
            'id': self.identifier,
            'tags': self.tags,
            'tokenization': tokenization,
            'content': self.content if self.output_config['output_content'] else '',
            'mentions': [m.json() for m in self.mentions if m._visible],
            'concepts': [c.json() for c in self.concepts if c._visible],
            'relations': [r.json() for r in self.relations],
            'mention_relations': [r.json() for r in self.mention_relations],
            'frames': []
        }


class BuilderConcept:

    def __init__(self, doc):
        self.doc = doc
        self.idx = -1
        self.mentions = []
        self.relations = []

    def add_mention(self, begin, end):
        mention = self.doc.add_mention(begin, end)
        mention.concept = self
        self.mentions.append(mention)

    def add_mention2(self, mention):
        mention.concept = self
        self.mentions.append(mention)

    def add_relation(self, relation):
        self.relations.append(relation)

    def is_visible(self):
        return len(self.get_tags()) > 0 or len(self.mentions) > 1 or len(self.relations) > 0 or self.doc.singletons

    def get_text(self):
        text = None
        for mention in self.mentions:
            text = mention.text if text is None or len(text) < len(mention.text) else text
        return text

    def get_tags(self):
        tags = set()
        for mention in self.mentions:
            tags.update(mention.tags)
        return tags

    def most_common(self, lst):
        # todo: !! in case of a tie, chose the one entity with most links in the corpus, right??
        return max(set(lst), key=lst.count)

    def get_link_gold(self):
        """

        :return: gets the link that repeats the most in the mentions of a particular concept.
        """
        return self.most_common([m.link_gold for m in self.mentions])

    def get_link_pred(self):
        """

        :return: gets the link that repeats the most in the mentions of a particular concept.
        """
        return self.most_common([m.link_pred for m in self.mentions])

    def json(self):
        return {
            'concept': self.idx,
            'text': self.get_text(),
            'count': len(self.mentions),
            'tags': list(self.get_tags()),
            'link_pred': self.get_link_pred()
        }


class BuilderMention:

    def __init__(self, doc, token_begin, token_end, coref_pointer=None):
        self.tags = set()
        self.token_begin = token_begin
        self.token_end = token_end
        self.char_begin = doc.begin[token_begin]
        self.char_end = doc.end[token_end]
        self.text = doc.content[self.char_begin:self.char_end]
        self.concept = None
        self.mention_relations = []
        self.idx = -1
        self.candidates = None
        self.scores = None  # these are linking scores
        self.coref_scores = None
        self.link_gold = None
        self.link_pred = None

        self.coref_connection_type = None
        self.coref_connection_pointer = None

        if coref_pointer is not None:
            self.coref_connection_type = coref_pointer['coref_connection_type']
            if 'mention' in self.coref_connection_type:
                doc_begin = doc.begin[coref_pointer['coref_connection_pointer'][0]]
                doc_end = doc.end[coref_pointer['coref_connection_pointer'][1]]
                self.coref_connection_pointer = {'begin': doc_begin,
                                                 'end': doc_end,
                                                 'text': doc.content[doc_begin:doc_end],
                                                 'score': coref_pointer['coref_connection_score']
                                                 }
            elif self.coref_connection_type == 'link':
                self.coref_connection_pointer = coref_pointer['coref_connection_pointer']

    def add_tag(self, tag):
        self.tags.add(tag)

    def add_mention_relation(self, relation):
        self.mention_relations.append(relation)

    def add_candidates_and_scores(self, candidates, scores):
        self.candidates = candidates
        self.scores = scores

    def add_coref_scores(self, coref_scores):
        self.coref_scores = coref_scores

    def add_link_gold(self, link):
        self.link_gold = link

    def add_link_pred(self, link):
        self.link_pred = link

    def json(self):
        out = {
            'concept': self.concept.idx,
            'begin': self.char_begin,
            'end': self.char_end,
            'text': self.text,
            'tags': list(self.tags)
        }
        if self.candidates is not None:
            out['candidates'] = self.candidates
            out['scores'] = self.scores

        if self.coref_scores is not None:
            out['coref_scores'] = self.coref_scores

        if self.link_gold is not None:
            out['link'] = self.link_gold
        if self.link_pred is not None:
            out['link_pred'] = self.link_pred

        if self.coref_connection_type is not None:
            out['coref_connection_type'] = self.coref_connection_type

        if self.coref_connection_pointer is not None:
            out['coref_connection_pointer'] = self.coref_connection_pointer
        return out


class BuilderRelation:

    def __init__(self, src, dst, rel):
        self.src = src
        self.dst = dst
        self.rel = rel

    def json(self):
        return {
            's': self.src.idx,
            'p': self.rel,
            'o': self.dst.idx
        }
