# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_autodiff_op.py
   Description :
   Author :       haxu
   date：          2018/3/4
-------------------------------------------------
   Change Activity:
                   2018/3/4:
-------------------------------------------------
"""
__author__ = 'haxu'

import thunder as th
import thunder.autodiff as ad
import numpy as np
import numpy.testing as npt


def test_dummy():
    assert 1 == 1


def test_identity():
    x2 = ad.Variable(name='x2')
    y = x2
    executor = ad.Executor([y])
    x2_val = 2 * np.ones(3)
    y_val, = executor.run(feed_shapes={x2: x2_val})
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)


if __name__ == '__main__':
    test_dummy()
    test_identity()
