import json
import torch
import torch.nn as nn
from datass.dictionary import Dictionary

def _padding(items, size, unknown):
    if len(items) < size:
        return items + [unknown] * (size - len(items))
    else:
        return items[:size]


# Create KB embeddings on the fly
class EntityEmbbederKB(nn.Module):

    def __init__(self, dictionaries, config):
        super(EntityEmbbederKB, self).__init__()
        self.dictionary = dictionaries[config['dictionary']].tolist()
        self.predicates = Dictionary()
        self.entities = Dictionary()
        self.p_padding = self.predicates.add('__PADDING__')
        self.e_padding = self.entities.add('__PADDING__')

        self.dim_pred = config['dim_predicates']
        self.dim_ents = config['dim_entities']
        self.dim_hidden = config['dim_hidden']
        self.max_facts = config['max_facts']

        self.load_triples(config['filename'])

        print("EntityEmbbederKB:", len(self.dictionary), len(self.preds))
        self.p_embedder = nn.Embedding(self.predicates.size, self.dim_pred)
        self.e_embedder = nn.Embedding(self.entities.size, self.dim_ents)
        self.network = nn.Sequential(
            nn.Linear(self.dim_pred+self.dim_ents, self.dim_hidden),
            nn.ReLU()
        )
        self.dim_output = self.dim_hidden

    def load_triples(self, filename):
        print("loading triples:", filename)
        maxsize = 0
        self.preds = {}
        self.objs = {}
        with open(filename, 'r') as file:
            for line in file.readlines():
                data = json.loads(line)
                identifier = data['id']
                idx = self.entities.add(identifier)
                maxsize = max(maxsize, len(data['pred']))
                self.preds[idx] = [self.predicates.add(x) for x in data['pred']]
                self.objs[idx] = [self.entities.add(x) for x in data['obj']]
        print("maximum size:", maxsize)

        nill_entity = self.entities.add('NILL')
        self.preds[nill_entity] = []
        self.objs[nill_entity] = []
    
    def forward(self, candidates):
        # print('candidates:', candidates.size())
        candidat2entity = { candidate_idx:self.entities.lookup(self.dictionary[candidate_idx]) for candidate_idx in set(candidates.view(-1).tolist()) }
        # print("number of entities:", len(candidat2entity))

        mapping2 = {}
        p_data = []
        e_data = []
        index = 0
        for candidate_idx, entity_idx in candidat2entity.items():
            if entity_idx not in self.preds:
                raise BaseException("no such entity:", entity_idx)
            p_data.append(_padding(self.preds[entity_idx], self.max_facts, self.p_padding))
            e_data.append(_padding(self.objs[entity_idx], self.max_facts, self.e_padding))
            mapping2[candidate_idx] = index
            index += 1
        p_vecs = torch.LongTensor(p_data).to(candidates.device)
        e_vecs = torch.LongTensor(e_data).to(candidates.device)

        inputs_x = torch.LongTensor([mapping2[candidate_idx] for candidate_idx in candidates.view(-1).tolist()]).view(candidates.size()).to(candidates.device)

        p_vecs = self.p_embedder(p_vecs)
        e_vecs = self.e_embedder(e_vecs)
        vecs = torch.cat((p_vecs, e_vecs), -1)
        vecs = self.network(vecs)
        vecs, _ = torch.max(vecs, 1)

        outputs = torch.index_select(vecs, 0, inputs_x.view(-1))
        outputs = outputs.view(candidates.size() + (-1,))
        # print('outputs:', outputs.size())

        return outputs