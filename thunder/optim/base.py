# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     base
   Description :
   Author :       haxu
   date：          2018/3/6
-------------------------------------------------
   Change Activity:
                   2018/3/6:
-------------------------------------------------
"""
__author__ = 'haxu'

import thunder.autodiff as ad

try:
    from thunder.ndarray import ndarray
except ImportError:
    pass


class Base:
    def __init__(self, cost, params, lr=0.1, use_gpu=False):
        self.cost = cost

        self.params = self._copy_to_gpu(params) if use_gpu else params
        self.lr = lr
        grads = ad.gradients(cost, params)
        grads.insert(0, cost)
        self.use_gpu = use_gpu
        self.executor = ad.Executor(grads, use_gpu=use_gpu)

    def step(self, feed_dict):
        raise NotImplementedError('This method should be implemented by subclasses')

    @staticmethod
    def _copy_to_gpu(params):
        ctx = ndarray.gpu(0)
        gpu_arrays = []
        for param in params:
            param.const = ndarray.array(param.const, ctx=ctx)
            gpu_arrays.append(param)
        return gpu_arrays
