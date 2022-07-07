import logging

import networkx as nx
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def mst(scores, lengths):
    T = torch.zeros(scores.size())
    for b, nodes in enumerate(lengths):
        if nodes > 0:
            s = scores[b, :, :].cpu().tolist()
            G = encode_graph(s, nodes)
            tree = nx.maximum_spanning_arborescence(G)
            decode_tree(T[b, :, :], tree)
    return T.to(scores.device)


def mst_with_tree(scores, lengths, mask):
    T = torch.zeros(scores.size())
    tree_lst = list()
    for b, nodes in enumerate(lengths):
        if nodes > 0:
            curr_mask = mask[b, :, :].cpu()
            # s = scores[b, :, :].cpu().tolist()
            # G = encode_graph_mtt(s, nodes, curr_mask)
            G = encode_graph_mtt_indexed(scores[b, :, :], curr_mask)
            tree = nx.maximum_spanning_arborescence(G)
            decode_tree(T[b, :, :], tree)
            tree_lst.append(tree)
    return T.to(scores.device), tree_lst


def mst_only_tree(scores, lengths, mask):
    tree_lst = list()
    for b, nodes in enumerate(lengths):
        if nodes > 0:
            curr_mask = mask[b, :, :].cpu()
            # s = scores[b, :, :].cpu().tolist()
            # G = encode_graph_mtt(s, nodes, curr_mask)
            G = encode_graph_mtt_indexed(scores[b, :, :], curr_mask)
            tree = nx.maximum_spanning_arborescence(G)
            # decode_tree(T[b, :, :], tree)
            tree_lst.append(tree)
    return tree_lst


def mst_only_branches(scores, lengths, mask):
    tree_lst = list()
    for b, nodes in enumerate(lengths):
        if nodes > 0:
            curr_mask = mask[b, :, :].cpu()
            # s = scores[b, :, :].cpu().tolist()
            # G = encode_graph_mtt(s, nodes, curr_mask)
            G = encode_graph_mtt_indexed(scores[b, :, :], curr_mask)
            tree = nx.maximum_branching(G)
            # decode_tree(T[b, :, :], tree)
            # my_branch_edges = list(nx.dfs_edges(tree))
            # print('branch edges: ', my_branch_edges)
            tree_lst.append(tree)
    return tree_lst


def encode_graph(scores, nodes):
    G = nx.DiGraph()
    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                G.add_edge(i + 1, j + 1, weight=scores[i][j])
    for i in range(nodes):
        G.add_edge(i + 1, 0, weight=scores[i][i])
    return G


def encode_graph_mtt(scores, nodes, mask):
    G = nx.DiGraph()

    for i in range(nodes):
        for j in range(nodes):
            if mask[i, j] == 1.0:
                G.add_edge(i, j, weight=scores[i][j])

    return G


def encode_graph_mtt_indexed(scores, mask):
    """
    The idea of this version is to access elements using indexing, theoretically making it faster
    :param scores:
    :param mask:
    :return:
    """
    G = nx.DiGraph()
    masked_indices = (mask == 1.0)
    masked_indices_nonzero = masked_indices.nonzero()
    masked_indices_nonzero = masked_indices_nonzero.tolist()
    masked_scores = scores[masked_indices].tolist()
    for pos, score in zip(masked_indices_nonzero, masked_scores):
        # vertex_from = int(pos[0].item())
        # vertex_to = int(pos[1].item())
        # edge_score = score.item()
        vertex_from = int(pos[0])
        vertex_to = int(pos[1])
        edge_score = score
        G.add_edge(vertex_from, vertex_to, weight=edge_score)

    return G


def decode_tree(output, tree):
    for src, dst in tree.edges():
        if dst == 0:
            # print(src, dst)
            logger.info('src: %s dst: %s' % (src, dst))
            output[src][src] = 1
        else:
            output[src][dst] = 1
    return output
