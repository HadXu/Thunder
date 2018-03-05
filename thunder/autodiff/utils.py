import numpy as np


def find_topo_sort(node_list):
    visited = set()
    topo_order = []
    for node in node_list:
        depth_first_search(node, visited, topo_order)
    return topo_order


def depth_first_search(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)