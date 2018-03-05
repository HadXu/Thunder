import numpy as np
from thunder.autodiff.autodiff import PlaceholderOp
from .utils import find_topo_sort

try:
    from thunder.ndarray import gpu_op, ndarray
except ImportError:
    pass


class Executor:
    def __init__(self, eval_list, use_gpu=False):
        self.eval_node_list = eval_list
        self.ctx = None
        if use_gpu:
            self.ctx = ndarray.gpu(0)

        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_arr_map = None
        self.node_to_shape_map = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in self.node_to_shape_map:
                continue

            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                print(node)
                self.node_to_shape_map[node] = node.const.shape
                continue

            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
            else:
                input_shpes = []
                for input_node in node.inputs:
                    input_shpes.append(self.node_to_shape_map[input_node])

                self.node_to_shape_map[node] = node.op.infer_shape(node, input_shpes)

    def memory_plan(self, feed_shapes):
        if self.node_to_arr_map is None:
            self.node_to_arr_map = {}

        for node in self.topo_order:
            if node in feed_shapes:
                continue
            self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)

    def run(self, feed_shapes, convert_to_numpy_ret_vals=False):
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_shapes.items():
            if use_numpy:
                assert isinstance(value, np.ndarray)
                node_to_val_map[node] = value
            else:
                if isinstance(value, np.ndarray):
                    node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"

        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        if not are_feed_shapes_equal(feed_shapes, self.feed_shapes):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            if not use_numpy:
                self.memory_plan(feed_shapes)

        for node in self.topo_order:
            if node in node_to_val_map:
                continue

            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                node_to_val_map[node] = node.const
                continue

            input_vals = [node_to_val_map[n] for n in node.inputs]

            if use_numpy:
                node_val = np.empty(shape=self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]

            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val

        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]

        return [node_to_val_map[n] for n in self.eval_node_list]

    @staticmethod
    def _are_feed_shapes_equal(sa, sb):
        if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
            return False
        unmatched_items = set(sa.items()) ^ set(sb.items())
        return len(unmatched_items)
