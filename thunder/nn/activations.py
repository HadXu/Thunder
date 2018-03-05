# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     activations
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
from thunder.autodiff.autodiff import Op
from thunder.nn.utils import softmax_func

try:
    from thunder.ndarray import gpu_op, ndarray
except ImportError:
    pass


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_grad(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = np.sign(np.maximum(input_vals[0], 0)) * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError('Gradient of ReluGradientOp not implemented')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Sigmoid({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = 0.5 + 0.5 * np.tanh(0.5 * input_vals[0])
        else:
            raise NotImplementedError('GPU version not yet implemented')

    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = sigmoid(x) - sigmoid(x) * sigmoid(x)
        return [g * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes)
        return input_shapes[0]


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'SoftmaxOp({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val)

    def gradient(self, node, output_grads):
        raise NotImplementedError('Not yet implemented, Please use CrossEntropy operator')

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


relu = ReluOp()
relu_grad = ReluGradientOp()
sigmoid = SigmoidOp()
softmax = SoftmaxOp()
