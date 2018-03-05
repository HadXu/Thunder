# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gradient
   Description :
   Author :       haxu
   date：          2018/3/5
-------------------------------------------------
   Change Activity:
                   2018/3/5:
-------------------------------------------------
"""
__author__ = 'haxu'

from .utils import sum_node_list
from .utils import find_topo_sort
from .autodiff import ones_like


# 反向求导
def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [ones_like(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad

        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
