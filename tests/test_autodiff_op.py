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
    x = ad.Variable(name='x')
    y = ad.Variable(name='y')

    z = x / y + 4

    z_grad_x, z_grad_y = ad.gradients(z, [x, y])
    executor = ad.Executor([z, z_grad_x, z_grad_y])
    x_val = 2 * np.ones(1)
    y_val = 3 * np.ones(1)
    z_val, z_grad_x, z_grad_y = executor.run(feed_shapes={x: x_val, y: y_val})
    print(z_val, z_grad_x, z_grad_y)


if __name__ == '__main__':
    test_dummy()
    test_identity()
