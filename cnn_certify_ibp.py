"""
cnn_certify_ibp.py

Certifies networks with IBP

Copyright (C) 2021, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Sijia Liu <liusiji5@msu.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""
import numpy as np
from numba import njit

from setup_mnist import MNIST
from setup_cifar import CIFAR

import tensorflow as tf
import time
import pickle
import random

linear_bounds = None

import statistics

part = 1

import sys

if len(sys.argv) > 1 and __name__ == '__main__':
    part = int(sys.argv[1])


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


# CNN model class
class CNNModel:
    def __init__(self, param_vals, strides, paddings, inp_shape=(28, 28, 1)):

        self.weights = []
        self.biases = []
        self.shapes = []
        self.pads = []
        self.strides = []
        self.sizes = []

        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        for (W, b), s, p in zip(param_vals, strides, paddings):

            print(cur_shape)
            if not np.isnan(W).any():  # Conv
                W = W.astype(np.float32)
                b = b.astype(np.float32)
                stride = (s, s)
                pad = (0, 0, 0, 0)  # p_hl, p_hr, p_wl, p_wr
                if p == 'SAME':
                    desired_h = int(np.ceil(cur_shape[0] / stride[0]))
                    desired_w = int(np.ceil(cur_shape[0] / stride[1]))
                    total_padding_h = stride[0] * (desired_h - 1) + W.shape[0] - cur_shape[0]
                    total_padding_w = stride[1] * (desired_w - 1) + W.shape[1] - cur_shape[1]
                    pad = (int(np.floor(total_padding_h / 2)), int(np.ceil(total_padding_h / 2)),
                           int(np.floor(total_padding_w / 2)), int(np.ceil(total_padding_w / 2)))
                cur_shape = (int((cur_shape[0] + pad[0] + pad[1] - W.shape[0]) / stride[0]) + 1,
                             int((cur_shape[1] + pad[2] + pad[3] - W.shape[1]) / stride[1]) + 1, W.shape[-1])
                self.strides.append(stride)
                self.pads.append(pad)
                self.sizes.append(None)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            else:  # Pooling
                print('pool')
                pool_size = W.shape
                stride = (s, s)
                pad = (0, 0, 0, 0)  # p_hl, p_hr, p_wl, p_wr
                if p == 'SAME':
                    desired_h = int(np.ceil(cur_shape[0] / stride[0]))
                    desired_w = int(np.ceil(cur_shape[0] / stride[1]))
                    total_padding_h = stride[0] * (desired_h - 1) + pool_size[0] - cur_shape[0]
                    total_padding_w = stride[1] * (desired_w - 1) + pool_size[1] - cur_shape[1]
                    pad = (int(np.floor(total_padding_h / 2)), int(np.ceil(total_padding_h / 2)),
                           int(np.floor(total_padding_w / 2)), int(np.ceil(total_padding_w / 2)))
                cur_shape = (int((cur_shape[0] + pad[0] + pad[1] - pool_size[0]) / stride[0]) + 1,
                             int((cur_shape[1] + pad[2] + pad[3] - pool_size[1]) / stride[1]) + 1, cur_shape[2])
                self.strides.append(stride)
                self.sizes.append(pool_size)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(np.full(pool_size + (1, 1), np.nan, dtype=np.float32))
                self.biases.append(np.full(1, np.nan, dtype=np.float32))

        for i in range(len(self.weights)):
            print(self.weights[i].shape)
            self.weights[i] = np.ascontiguousarray(self.weights[i].transpose((3, 0, 1, 2)).astype(np.float32))
            self.biases[i] = np.ascontiguousarray(self.biases[i].astype(np.float32))


@njit
def conv(W, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((int((x.shape[0] - W.shape[1] + p_hl + p_hr) / s_h) + 1,
                  int((x.shape[1] - W.shape[2] + p_wl + p_wr) / s_w) + 1, W.shape[0]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            if 0 <= s_h * a + i - p_hl < x.shape[0] and 0 <= s_w * b + j - p_wl < x.shape[1]:
                                y[a, b, c] += W[c, i, j, k] * x[s_h * a + i - p_hl, s_w * b + j - p_wl, k]
    return y


@njit
def pool(pool_size, x0, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y0 = np.zeros((int((x0.shape[0] + p_hl + p_hr - pool_size[0]) / s_h) + 1,
                   int((x0.shape[1] + p_wl + p_wr - pool_size[1]) / s_w) + 1, x0.shape[2]), dtype=np.float32)
    for x in range(y0.shape[0]):
        for y in range(y0.shape[1]):
            for r in range(y0.shape[2]):
                cropped = x0[s_h * x - p_hl:pool_size[0] + s_h * x - p_hl, s_w * y - p_wl:pool_size[1] + s_w * y - p_wl,
                          r]
                y0[x, y, r] = cropped.max()
    return y0


# Main function to find output bounds
def find_output_bounds(weights, biases, shapes, pads, strides, x0, act, L, U):
    for i in range(len(weights)):
        if i != 0:
            U = act(U)
            L = act(L)
        if not np.isnan(weights[i]).any():  # Conv
            U_new = conv(np.maximum(weights[i], 0), U, np.asarray(pads[i]), np.asarray(strides[i])) + \
                    conv(np.minimum(weights[i], 0), L, np.asarray(pads[i]), np.asarray(strides[i])) + biases[i]
            L_new = conv(np.maximum(weights[i], 0), L, np.asarray(pads[i]), np.asarray(strides[i])) + \
                    conv(np.minimum(weights[i], 0), U, np.asarray(pads[i]), np.asarray(strides[i])) + biases[i]
            U, L = U_new, L_new
    return L, U


# Warms up numba functions
def warmup(model, x, eps_0, p_n, fn):
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0, :, :, :]).reshape([1] + list(W.shape[1:])), dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1, 1, 1))
    fn(weights, biases, shapes, model.pads, model.strides, x, eps_0, p_n)


# Runs CNN with input x
def run_inp(weights, biases, shapes, pads, strides, sizes, x, act):
    for i in range(len(weights)):
        if i != 0:
            x = act(x)
        if not np.isnan(weights[i]).any():  # Conv
            x = conv(weights[i], x, np.asarray(pads[i]), np.asarray(strides[i])) + biases[i]
        else:
            x = pool(np.asarray(sizes[i]), x, np.asarray(pads[i]), np.asarray(strides[i]))
    return x


def sample_std(lst, samples=100, trials=100):
    means = []
    for i in range(trials):
        means.append(sum(random.sample(lst, samples)) / samples)
    return statistics.stdev(means)


def certify(network, strides, paddings, epss, n_pts=None, test=True, cifar=False, tinyimagenet=False, normalize=False,
            act='relu'):
    if cifar:
        data = CIFAR()
    elif tinyimagenet:
        data = tinyImagenet()
    else:
        data = MNIST()
    if test:
        x_val = data.test_data + 0.5
        y_val = data.test_labels
        if cifar and normalize:  # normalize
            x_val = (x_val - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])
    else:
        x_val = data.validation_data + 0.5
        y_val = data.validation_labels
        if cifar and normalize:  # normalize
            x_val = (x_val - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])

    np.random.seed(99)
    if n_pts is None:
        n_pts = x_val.shape[0]  # Full test set
    idx = np.random.permutation(np.arange(x_val.shape[0]))[:n_pts]
    x_val = x_val[idx, :, :, :]
    y_val = y_val[idx, :]
    vals = []
    for i in range(n_pts):
        vals.append((np.float32(x_val[i, :, :, :]), int(np.argmax(y_val[i, :]))))
    tests = vals

    print('Loading model')
    with open('networks/' + network + '.pkl', 'rb') as file:
        param_vals = pickle.load(file)
    if cifar:
        model = CNNModel(param_vals, strides, paddings, inp_shape=(32, 32, 3))
    else:
        model = CNNModel(param_vals, strides, paddings, inp_shape=(28, 28, 1))

    if act == 'relu':
        act_fn = lambda x: np.maximum(x, 0)
    elif act == 'tanh':
        act_fn = np.tanh

    results = []
    full = []
    for test in tests:
        image = test[0]
        true_label = test[1]
        out = run_inp(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes,
                      image.astype(np.float32), act_fn)
        predict_label = np.argmax(np.squeeze(out))
        if int(predict_label) == int(true_label):
            full.append(1)
        else:
            full.append(0)
    results.append(sum(full) / len(tests))
    print(results)
    print(sample_std(full))

    start_time = time.time()
    print("Network = {}".format(network))
    eps_results = [0 for i in epss]
    eps_full = [[] for i in epss]
    for test_num, test in enumerate(tests):
        image = test[0]
        true_label = test[1]
        out = run_inp(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes,
                      image.astype(np.float32), act_fn)
        predict_label = np.argmax(np.squeeze(out))
        print("[L1] num = {}, predict_label = {}, true_label = {}".format(test_num, predict_label, true_label))

        if int(predict_label) == int(true_label):
            # Search over epsilon
            for i, eps in enumerate(epss):
                print("[L2] eps = {}".format(eps))
                if normalize:
                    U = image + eps / np.asarray([0.2023, 0.1994, 0.2010])
                    L = image - eps / np.asarray([0.2023, 0.1994, 0.2010])
                    U = np.clip(U, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                                (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
                    L = np.clip(L, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                                (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
                else:
                    U = np.clip(image + eps, 0, 1)
                    L = np.clip(image - eps, 0, 1)

                failed = False
                for t in range(10):
                    if t != predict_label:
                        target_label = t
                        weights = model.weights[:-1]
                        biases = model.biases[:-1]
                        shapes = model.shapes[:-1]
                        W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
                        last_weight = (W[predict_label, :, :, :] - W[target_label, :, :, :]).reshape(
                            [1] + list(W.shape[1:]))
                        weights.append(last_weight)
                        biases.append(np.asarray([b[predict_label] - b[target_label]]))
                        shapes.append((1, 1, 1))
                        LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides,
                                                    image.astype(np.float32), act_fn, L.astype(np.float32),
                                                    U.astype(np.float32))
                        # print("[L3] LB = {}, UB = {}, t = {}".format(LB, UB, t))
                        if LB < 0:
                            failed = True
                            break
                if not failed:
                    eps_results[i] += 1
                    eps_full[i].append(1)
                else:
                    for j in range(i, len(epss)):
                        eps_full[j].append(0)
                    break
        else:
            for i, eps in enumerate(epss):
                eps_full[i].append(0)
        print([sum(full[:test_num + 1]) / (test_num + 1)] + [r / (test_num + 1) for r in
                                                             eps_results])  # print current average statistics
        if test_num >= 100:
            print([sample_std(full[:test_num + 1])] + [sample_std(l) for l in eps_full])  # print current std statistics
        print(time.time() - start_time)
    results = results + [correct / len(tests) for correct in eps_results]
    return results
