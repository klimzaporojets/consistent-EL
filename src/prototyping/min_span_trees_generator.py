import itertools
from typing import Set, List

import torch


def get_all_span_trees(nodes_left: List, root_node: str, edges: Set, connected_nodes: Set, nr_all_nodes,
                       possible_combinations: Set):
    left_nodes_combinations = [list(i) for i in itertools.product([1, 0], repeat=len(nodes_left))]
    # local_connected_nodes = connected_nodes.copy()
    for curr_ln_comb in left_nodes_combinations:
        edges_added: List = list()
        new_left_nodes = list()
        for idx, curr_node in enumerate(nodes_left):
            # if curr_ln_comb[idx] == 1 and curr_node not in local_connected_nodes:
            if curr_ln_comb[idx] == 1 and curr_node not in connected_nodes:
                assert (root_node, curr_node) not in edges
                edges.add((root_node, curr_node))
                edges_added.append((root_node, curr_node))
                connected_nodes.add(curr_node)
                if len(connected_nodes) == nr_all_nodes:
                    edges_to_append = sorted(list(edges), key=lambda x: (x[0], x[1]))
                    edges_to_append = tuple(edges_to_append)
                    # possible_combinations.add(edges.copy())
                    possible_combinations.add(edges_to_append)
                    # break
            else:
                new_left_nodes.append(curr_node)

        # assert len(new_left_nodes) < len(left_nodes)
        if len(connected_nodes) < nr_all_nodes and len(new_left_nodes) < len(nodes_left):
            assert len(new_left_nodes) == nr_all_nodes - len(connected_nodes)
            assert len(new_left_nodes) > 0
            for curr_added_edge in edges_added:
                get_all_span_trees(new_left_nodes, curr_added_edge[1], edges, connected_nodes, nr_all_nodes,
                                   possible_combinations)

        for curr_added_edge in edges_added:
            edges.remove(curr_added_edge)
            connected_nodes.remove(curr_added_edge[1])


def expand_this(root: str, binary_mask: List, nodes_left: List, edges: Set, connected_nodes: Set):
    new_nodes_left = list()
    edges_expanded = set()
    for idx, curr_node_left in nodes_left:
        if binary_mask[idx] == 1 and curr_node_left not in connected_nodes:
            edge = (root, curr_node_left)
            edges_expanded.add(edge)
            assert edge not in edges
            edges.add(edge)
        else:
            new_nodes_left.append(curr_node_left)
    return new_nodes_left, edges_expanded


def contract_this(edges_expanded: Set, edges: Set, connected_nodes: Set):
    for curr_edge_expanded in edges_expanded:
        edges.remove(curr_edge_expanded)
        connected_nodes.remove(curr_edge_expanded[1])


def get_all_span_trees_v2(nodes_left: List, root_node: str, edges: Set, connected_nodes: Set, nr_all_nodes,
                          possible_combinations: Set):
    pointer_nodes = [root_node]
    for curr_pointer_node in pointer_nodes:
        left_nodes_combinations = [list(i) for i in itertools.product([1, 0], repeat=len(nodes_left))]
        # TODO: got stuck here, look now working on get_all_span_trees_v3 on adjacency matrix approach


# def get_all_span_trees_v3():
def get_all_span_trees_v3(nr_gold_mentions: int = 5):
    # a = [0, 1, 2, 3, 4, 5]
    # b = [0, 1, 2, 3, 4, 5]
    # c = [0, 1, 2, 3, 4, 5]
    # d = [0, 1, 2, 3, 4, 5]
    # e = [0, 1, 2, 3, 4, 5]
    #
    # idx_to_node = ['R', 'A', 'B', 'C', 'D', 'E']
    # node_to_idx = {'R': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    # tensor_a = torch.tensor(a)
    # tensor_b = torch.tensor(b)
    # tensor_c = torch.tensor(c)
    # tensor_d = torch.tensor(d)
    # tensor_e = torch.tensor(e)

    # cart_prod_idx = torch.cartesian_prod(tensor_a, tensor_b, tensor_c, tensor_d, tensor_e)

    node_letter_names = 'ABCDEFGHIJKLMNOPQRSTUVWXUZ'

    idx_to_node = ['R']
    node_to_idx = {'R': 0}

    for id in range(1, nr_gold_mentions + 1):
        idx_to_node.append(node_letter_names[id - 1])
        node_to_idx[node_letter_names[id - 1]] = id

    tensors_mention_idxs = [torch.tensor(list(range(nr_gold_mentions + 1))) for _ in range(nr_gold_mentions)]
    cart_prod_idx = torch.cartesian_prod(*tensors_mention_idxs)

    adjacency = torch.zeros(cart_prod_idx.size(0), cart_prod_idx.size(1), cart_prod_idx.size(1) + 1)

    cart_prod_idx2 = cart_prod_idx.reshape(-1) + torch.arange(
        cart_prod_idx.size(0) * cart_prod_idx.size(1)) * adjacency.size(2)

    adjacency.view(-1)[cart_prod_idx2] = 1

    # concatenates the row to root with all zeros, since the root doesn't get edges pointing to it
    empty_row_to_concat = torch.zeros(adjacency.size(0), 1, adjacency.size(2))
    adjacency = torch.cat([empty_row_to_concat, adjacency], dim=1)
    # now counts the unique spanning trees
    nr_spanning_trees = 0
    print('adjacency shape: ', adjacency.shape)
    nr_processed = 0
    for curr_adj_matrix in adjacency:
        nr_processed += 1
        if nr_processed % 1000 == 0:
            print('nr pre-processed: ', nr_processed, '  nr_kept (spanning trees without loops): ', nr_spanning_trees)
        # continue if the root doesn't have outbound relations
        if curr_adj_matrix[:, 0].sum() == 0:
            # print('continuing because no connection from root: ')
            # print(curr_adj_matrix)
            continue
        # continue if there are relations to itself (main diagonal)
        if torch.diagonal(curr_adj_matrix, 0).sum() > 0:
            # print('continuing because of elements self-referencing main diagonal: ')
            # print(curr_adj_matrix)
            continue

        # continue if symmetries detected (ex: A -> B ; B -> A
        upper_triangle = torch.triu(curr_adj_matrix)
        lower_triangle = torch.tril(curr_adj_matrix)
        lower_triangle_t = lower_triangle.transpose(1, 0)
        symm_check = upper_triangle + lower_triangle_t
        if torch.max(symm_check) > 1:
            # print('continuing because of symmetries detected')
            # print(curr_adj_matrix)
            continue

        # continue if a larger cycle is found such as in (R, A, B, C, D): R->A ; B->C; C->D; D->B
        # use both trace and brute force to detect this, details on:
        # https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix#:~:text=Start%20from%20an%20edge%20(i,then%20a%20cycle%20is%20detected
        # check if from the root all the nodes can be accessed using tree traversal
        traversed_root = (curr_adj_matrix[:, 0] == 1).nonzero()
        traversed_nodes = set()
        for idx_root in traversed_root:
            traversed_nodes.add(idx_to_node[idx_root[0]])

        def recursive_tree_traverse(traversed_nodes: Set, to_traverse_nodes: Set):
            curr_to_traverse_to = set()
            for to_traverse in to_traverse_nodes:
                idx = node_to_idx[to_traverse]
                pointed_to = (curr_adj_matrix[:, idx] == 1).nonzero()
                for curr_pointed_node_idx in pointed_to:
                    curr_pointed_node = idx_to_node[curr_pointed_node_idx[0]]
                    if curr_pointed_node in traversed_nodes:
                        return False  # loop detected!
                    else:
                        traversed_nodes.add(curr_pointed_node)
                        curr_to_traverse_to.add(curr_pointed_node)
            if len(curr_to_traverse_to) > 0:
                return recursive_tree_traverse(traversed_nodes, curr_to_traverse_to)
            return True

        recursive_tree_traverse(traversed_nodes, to_traverse_nodes=traversed_nodes.copy())
        if len(traversed_nodes) < len(idx_to_node) - 1:
            continue

        nr_spanning_trees += 1
        # print('valid spanning tree: ')
        # print(curr_adj_matrix)
    print('total number of distinct spanning trees calculated: ', nr_spanning_trees)


if __name__ == "__main__":
    # get_all_span_trees_v3(7)
    get_all_span_trees_v3(5)

    exit(0)
    root_node = 'A'

    edges = set()

    curr_root = root_node
    connected_nodes = {root_node}
    left_nodes = ['B', 'C', 'D', 'E']

    left_nodes_combinations = [list(i) for i in itertools.product([0, 1], repeat=len(left_nodes))]

    possible_combinations = set()
    get_all_span_trees(left_nodes, root_node, edges, connected_nodes, len(left_nodes) + 1, possible_combinations)
    # get_min_span_trees(left_nodes, root_node, edges, connected_nodes, len(nodes) + 1, possible_combinations)
    print('possible combinations: ', possible_combinations)
    print('len possible combinations: ', len(possible_combinations))
