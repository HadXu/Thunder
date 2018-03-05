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

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub(self, other)
        else:
            return sub_const(self, other)

    def __rsub__(self, other):
        return ref_sub_const(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            return mul_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            return div_const(self, other)

    __radd__ = __add__
    __rmul__ = __mul__


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


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}-{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] - input_vals[1]
        else:
            gpu_op.matrix_elementwise_subtract(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grads):
        return [output_grads, -1 * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class SubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}-{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] - node.const
        else:
            gpu_op.matrix_elementwise_subtract_by_const(input_vals[0], node.const, output_val)

    def gradient(self, node, output_grads):
        return [output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReflectedSubByConstOp(Op):
    """const - variable"""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:f}-{1:s})'.format(const_val, node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        return node.const - input_vals[0]

    def gradient(self, node, output_grads):
        return [-1 * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}*{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] * input_vals[1]
        else:
            ip_1_shape = input_vals[0].shape
            ip_2_shape = input_vals[1].shape
            if ip_1_shape == ip_2_shape:
                gpu_op.matrix_elementwise_multiply(input_vals[0], input_vals[1], output_val)
            elif ip_1_shape == (1,):
                const_val = input_vals[0].asnumpy()[0]
                gpu_op.matrix_elementwise_multiply_by_const(input_vals[1], const_val, output_val)
            elif ip_2_shape == (1,):
                const_val = input_vals[1].asnumpy()[0]
                gpu_op.matrix_elementwise_multiply_by_const(input_vals[0], const_val, output_val)
            else:
                pass  # TODO (upul) handle ip_1_shape != ip_2_shape

    def gradient(self, node, output_grads):
        return [node.inputs[1] * output_grads, node.inputs[0] * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        if input_shapes[0] == (1,):
            return input_shapes[1]
        elif input_shapes[1] == (1,):
            return input_shapes[0]
        elif input_shapes[0] == input_shapes[1]:
            return input_shapes[0]
        else:
            stmt = 'Invalid dimensions {0:s}, (1:s)'.format(input_shapes[0], input_shapes[1])
            raise RuntimeError(stmt)


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}*{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = node.const * input_vals[0]
        else:
            gpu_op.matrix_elementwise_multiply_by_const(
                input_vals[0], node.const, output_val)

    def gradient(self, node, output_grads):
        return [node.const * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.trans_A = trans_A
        new_node.trans_B = trans_B
        new_node.name = 'MatMul({0:s}, {1:s}'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            if node.trans_A:
                input_vals[0] = input_vals[0].T
            if node.trans_B:
                input_vals[1] = input_vals[1].T
            output_val[:] = np.dot(input_vals[0], input_vals[1])
        else:
            gpu_op.matrix_multiply(
                input_vals[0], node.trans_A,
                input_vals[1], node.trans_B,
                output_val)

    def gradient(self, node, output_grads):
        grad_A = matmul(output_grads, node.inputs[1], trans_A=False, trans_B=True)
        grad_B = matmul(node.inputs[0], output_grads, trans_A=True, trans_B=False)
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 2
        (row_A, col_A) = input_shapes[0]
        if node.trans_A:
            row_A, col_A = col_A, row_A
        (row_B, col_B) = input_shapes[1]
        if node.trans_B:
            row_B, col_B = col_B, row_B

        assert col_A == row_B
        return (row_A, col_B)


class DivOp(Op):
    def __call__(self, nodeA, nodeB):
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA, nodeB]
        new_node.name = '({0:s}/{1:s})'.format(nodeA.name, nodeB.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] / input_vals[1]
        else:
            gpu_op.matrix_elementwise_division(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grads):
        grad_A = output_grads / node.inputs[1]
        grad_B = -1.0 * output_grads * node.inputs[0] / (node.inputs[1] * node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}/{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] / node.const
        else:
            gpu_op.matrix_elementwise_div_by_const(input_vals[0], node.const, output_val)

    def gradient(self, node, output_grads):
        return [output_grads / node.const]

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

    def infer_shape(self, node, input_shapes):
        pass


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

sub = SubOp()
sub_const = SubByConstOp()
ref_sub_const = ReflectedSubByConstOp()

mul = MulOp()
mul_const = MulByConstOp()
matmul = MatMulOp()

div = DivOp()
div_const = DivByConstOp()

placeholder = PlaceholderOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
