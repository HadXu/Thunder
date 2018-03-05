import numpy as np

try:
    from thunder.ndarray import gpu_op, darray
except ImportError:
    pass


class Node(object):
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const = None
        self.name = ""

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            return add_const(self, other)

    __radd__ = __add__


class Op(object):
    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        raise NotImplementedError

    def gradient(self, node, output_grads):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, nodeA, nodeB):
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA, nodeB]
        new_node.name = '({}+{})'.format(nodeA.name, nodeB.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] + input_vals[1]
        else:
            pass

    def gradient(self, node, output_grads):

        return [output_grads, output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const = const_val
        new_node.inputs = [node_A]
        new_node.name = '({0:s}+{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = node.const + input_vals[0]
        else:
            pass

    def gradient(self, node, output_grads):
        return [output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class PlaceholderOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        return None


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Zeroslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert isinstance(input_vals[0], np.ndarray)
            output_val[:] = np.zeros(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 0)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        if input_shapes[0] == 1:  # TODO (upul) do we need this if ?
            return (1,)
        else:
            return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Oneslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert isinstance(input_vals[0], np.ndarray)
            output_val[:] = np.ones(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 1)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        if input_shapes[0] == 1:  # TODO (upul) do we need this if ?
            return (1,)
        else:
            return input_shapes[0]


def Variable(name):
    placeholder_node = placeholder()
    placeholder_node.name = name
    return placeholder_node


def Parameter(name, init):
    parameter_node = placeholder()
    parameter_node.name = name
    parameter_node.const = init
    return parameter_node


add = AddOp()
add_const = AddByConstOp()
placeholder = PlaceholderOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
