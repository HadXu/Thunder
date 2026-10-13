# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     loss_functions
   Description :
   Author :       haxu
   date：          2018/3/5
-------------------------------------------------
   Change Activity:
                   2018/3/5:
-------------------------------------------------
"""
__author__ = 'haxu'

import numpy as np
from thunder.autodiff.autodiff import Op, zeros_like
from .utils import log_sum_exp, softmax_func

try:
    from thunder.ndarray import gpu_op, ndarray
except ImportError:
    pass


class CrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'CrossEntropy({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            logits = input_vals[0]
            actual = input_vals[1]
            safe_log_softmax = logits - log_sum_exp(logits)
            output_val[:] = np.mean(-np.sum(actual * safe_log_softmax, axis=1), keepdims=True)
        else:
            gpu_op.softmax_cross_entropy(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grads):
        """https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy"""
        grad_A = cross_entropy_grad(node.inputs[0], node.inputs[1], output_grads)
        grad_B = zeros_like(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return 1,


class CrossEntropyGradientOp(Op):
    """(softmax(logits) - labels) / batch_size * output_grad

    The loss is averaged over the batch, so the gradient must be divided by
    the batch size, which is only known at run time.
    """

    def __call__(self, node_A, node_B, node_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_grad]
        new_node.name = 'CrossEntropyGradient({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        logits, actual, output_grad = input_vals
        batch_size = logits.shape[0]
        if use_numpy:
            output_val[:] = (softmax_func(logits) - actual) / batch_size * output_grad
        else:
            gpu_op.softmax(logits, output_val)
            gpu_op.matrix_elementwise_subtract(output_val, actual, output_val)
            scale = output_grad.asnumpy()[0] / batch_size
            gpu_op.matrix_elementwise_multiply_by_const(output_val, scale, output_val)

    def gradient(self, node, output_grads):
        raise NotImplementedError('Gradient of CrossEntropyGradientOp not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


softmax_cross_entropy_with_logits = CrossEntropyOp()
cross_entropy_grad = CrossEntropyGradientOp()
