# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
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


def softmax_func(x):
    stable_values = x - np.max(x, axis=1, keepdims=True)
    return np.exp(stable_values) / np.sum(np.exp(stable_values), axis=1, keepdims=True)


def log_sum_exp(x):
    mx = np.max(x, axis=1, keepdims=True)
    safe = x - mx
    return mx + np.log(np.sum(np.exp(safe), axis=1, keepdims=True))
