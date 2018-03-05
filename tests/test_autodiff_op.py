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

    z = x / y + x / 2. + 4

    z_grad_x, z_grad_y = ad.gradients(z, [x, y])
    executor = ad.Executor([z, z_grad_x, z_grad_y])
    x_val = 2 * np.ones(1)
    y_val = 3 * np.ones(1)
    z_val, z_grad_x, z_grad_y = executor.run(feed_shapes={x: x_val, y: y_val})
    print(z_val, z_grad_x, z_grad_y)


def test_matmul():
    x = ad.Variable(name='x')
    y = ad.Variable(name='y')

    z = ad.matmul(x, y)

    z_grad_x, z_grad_y = ad.gradients(z, [x, y])

    executor = ad.Executor([z, z_grad_x, z_grad_y])

    x_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    y_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

    z_val, z_grad_x, z_grad_y = executor.run(feed_shapes={x: x_val, y: y_val})

    expected_yval = np.matmul(x_val, y_val)

    expected_grad_x_val = np.matmul(np.ones_like(expected_yval), np.transpose(y_val))
    expected_grad_y_val = np.matmul(x_val.T, np.ones_like(expected_yval))

    assert np.array_equal(expected_yval, z_val)
    assert np.array_equal(expected_grad_x_val, z_grad_x)
    assert np.array_equal(expected_grad_y_val, z_grad_y)


def test_relu():
    x = ad.Variable(name='x')
    y = th.nn.relu(x)

    grad_x2, = ad.gradients(y, [x])

    executor = ad.Executor([y, grad_x2])
    x_val = np.array([[-1, 2, 3], [1, -2, 0]])
    y_val, grad_x2_val = executor.run(feed_shapes={x: x_val})
    expected_y_val = np.array([[0, 2, 3], [1, 0, 0]])
    expected_x2_grad = np.array([[0, 1, 1], [1, 0, 0]])

    assert np.array_equal(y_val, expected_y_val)
    assert np.array_equal(grad_x2_val, expected_x2_grad)


def test_sigmoid():
    x = ad.Variable(name='x')
    y = th.nn.sigmoid(x)

    grad_x, = ad.gradients(y, [x])

    executor = ad.Executor([y, grad_x])

    x_val = np.array([-100, 0, 100])

    y_val, grad_x_val = executor.run(feed_shapes={x: x_val})

    print(y_val, grad_x_val)

def test_softmax():
    x2_pred = ad.Variable(name='x2_pred')
    x2_actu = ad.Variable(name='x2_actu')

    y = th.nn.softmax_cross_entropy_with_logits(x2_pred, x2_actu)
    x2_pred_grad, x2_actu_grad = ad.gradients(y, [x2_pred, x2_actu])
    x2_pred_val = np.array([[0.8, 0.01, 0.5], [0.8, 0.01, 0.5]])
    x2_actu_val = np.array([[1.0, 1.0, 0], [1.0, 1.0, 0]])

    executor = ad.Executor([y, x2_pred_grad, x2_actu_grad])
    y_val, x2_pred_grad_val, x2_actu_grad_val = executor.run(feed_shapes={x2_pred: x2_pred_val, x2_actu: x2_actu_val})

    print(x2_pred_grad_val)


if __name__ == '__main__':
    test_dummy()
    test_identity()
    test_matmul()
    test_relu()
    test_sigmoid()
    test_softmax()
