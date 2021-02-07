"""
cnn_certify.py

Certifies networks with linear bounding certifiers

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

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import time
from activations import relu_linear_bounds, ada_linear_bounds, zero_linear_bounds
from cnn_certify_ibp import find_output_bounds as find_output_bounds_ibp

linear_bounds = None


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
            # print(cur_shape)
            if not np.isnan(W).any() and type(s) is not str:  # Conv
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
            elif type(s) is str:  # Residual
                s = int(s[1:])
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
                self.sizes.append('res')
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


# @njit
def conv_bound(W, b, pad, stride, x0, L, U, p_n):
    if p_n == 105:
        W_minus = np.minimum(W, 0.0).astype(np.float32)
        W_plus = np.maximum(W, 0.0).astype(np.float32)
        UB = conv(W_plus, U, pad, stride) + conv(W_minus, L, pad, stride) + b
        LB = conv(W_plus, L, pad, stride) + conv(W_minus, U, pad, stride) + b
    return LB, UB


@njit
def conv_full(A, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(A.shape[3]):
                    for j in range(A.shape[4]):
                        for k in range(A.shape[5]):
                            if 0 <= s_h * a + i - p_hl < x.shape[0] and 0 <= s_w * b + j - p_wl < x.shape[1]:
                                y[a, b, c] += A[a, b, c, i, j, k] * x[s_h * a + i - p_hl, s_w * b + j - p_wl, k]
    return y


@njit
def conv_bound_full(A, B, pad, stride, x0, L, U, p_n):
    if p_n == 105:
        A_minus = np.minimum(A, 0.0).astype(np.float32)
        A_plus = np.maximum(A, 0.0).astype(np.float32)
        UB = conv_full(A_plus, U, pad, stride) + conv_full(A_minus, L, pad, stride) + B
        LB = conv_full(A_plus, L, pad, stride) + conv_full(A_minus, U, pad, stride) + B
    return LB, UB


@njit
def upper_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0] * (A.shape[3] - 1) + W.shape[1],
                      inner_stride[1] * (A.shape[4] - 1) + W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    assert A.shape[5] == W.shape[0]

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0 <= t + stride[0] * inner_stride[0] * x - inner_stride[0] * pad[0] - inner_pad[0] < inner_shape[
                        0] and 0 <= u + stride[1] * inner_stride[1] * y - inner_stride[1] * pad[2] - inner_pad[2] < \
                            inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0 <= t - inner_stride[0] * p < W.shape[1] and 0 <= u - inner_stride[1] * q < W.shape[
                                    2] and 0 <= p + stride[0] * x - pad[0] < alpha_u.shape[0] and 0 <= q + stride[
                                    1] * y - pad[2] < alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x, y, z, t, u, v] += W[r, t - inner_stride[0] * p, u -
                                                                             inner_stride[1] * q, v] * alpha_u[
                                                                               p + stride[0] * x - pad[0], q + stride[
                                                                                   1] * y - pad[2], r] * A_plus[
                                                                               x, y, z, p, q, r]
                                                A_new[x, y, z, t, u, v] += W[r, t - inner_stride[0] * p, u -
                                                                             inner_stride[1] * q, v] * alpha_l[
                                                                               p + stride[0] * x - pad[0], q + stride[
                                                                                   1] * y - pad[2], r] * A_minus[
                                                                               x, y, z, p, q, r]

    B_new = conv_full(A_plus, alpha_u * b + beta_u, pad, stride) + conv_full(A_minus, alpha_l * b + beta_l, pad,
                                                                             stride) + B
    return A_new, B_new


@njit
def lower_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0] * (A.shape[3] - 1) + W.shape[1],
                      inner_stride[1] * (A.shape[4] - 1) + W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    assert A.shape[5] == W.shape[0]
    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0 <= t + stride[0] * inner_stride[0] * x - inner_stride[0] * pad[0] - inner_pad[0] < inner_shape[
                        0] and 0 <= u + stride[1] * inner_stride[1] * y - inner_stride[1] * pad[2] - inner_pad[2] < \
                            inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0 <= t - inner_stride[0] * p < W.shape[1] and 0 <= u - inner_stride[1] * q < W.shape[
                                    2] and 0 <= p + stride[0] * x - pad[0] < alpha_u.shape[0] and 0 <= q + stride[
                                    1] * y - pad[2] < alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x, y, z, t, u, v] += W[r, t - inner_stride[0] * p, u -
                                                                             inner_stride[1] * q, v] * alpha_l[
                                                                               p + stride[0] * x - pad[0], q + stride[
                                                                                   1] * y - pad[2], r] * A_plus[
                                                                               x, y, z, p, q, r]
                                                A_new[x, y, z, t, u, v] += W[r, t - inner_stride[0] * p, u -
                                                                             inner_stride[1] * q, v] * alpha_u[
                                                                               p + stride[0] * x - pad[0], q + stride[
                                                                                   1] * y - pad[2], r] * A_minus[
                                                                               x, y, z, p, q, r]
    B_new = conv_full(A_plus, alpha_l * b + beta_l, pad, stride) + conv_full(A_minus, alpha_u * b + beta_u, pad,
                                                                             stride) + B
    return A_new, B_new


@njit
def pool_linear_bounds(LB, UB, pad, stride, pool_size):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    alpha_u = np.zeros((pool_size[0], pool_size[1], int((UB.shape[0] + p_hl + p_hr - pool_size[0]) / s_h) + 1,
                        int((UB.shape[1] + p_wl + p_wr - pool_size[1]) / s_w) + 1, UB.shape[2]), dtype=np.float32)
    beta_u = np.zeros((int((UB.shape[0] + p_hl + p_hr - pool_size[0]) / s_h) + 1,
                       int((UB.shape[1] + p_wl + p_wr - pool_size[1]) / s_w) + 1, UB.shape[2]), dtype=np.float32)
    alpha_l = np.zeros((pool_size[0], pool_size[1], int((LB.shape[0] + p_hl + p_hr - pool_size[0]) / s_h) + 1,
                        int((LB.shape[1] + p_wl + p_wr - pool_size[1]) / s_w) + 1, LB.shape[2]), dtype=np.float32)
    beta_l = np.zeros((int((LB.shape[0] + p_hl + p_hr - pool_size[0]) / s_h) + 1,
                       int((LB.shape[1] + p_wl + p_wr - pool_size[1]) / s_w) + 1, LB.shape[2]), dtype=np.float32)

    for x in range(alpha_u.shape[2]):
        for y in range(alpha_u.shape[3]):
            for r in range(alpha_u.shape[4]):
                cropped_LB = LB[s_h * x - p_hl:pool_size[0] + s_h * x - p_hl,
                             s_w * y - p_wl:pool_size[1] + s_w * y - p_wl, r]
                cropped_UB = UB[s_h * x - p_hl:pool_size[0] + s_h * x - p_hl,
                             s_w * y - p_wl:pool_size[1] + s_w * y - p_wl, r]

                max_LB = cropped_LB.max()
                idx = np.where(cropped_UB >= max_LB)
                u_s = np.zeros(len(idx[0]), dtype=np.float32)
                l_s = np.zeros(len(idx[0]), dtype=np.float32)
                gamma = np.inf
                for i in range(len(idx[0])):
                    l_s[i] = cropped_LB[idx[0][i], idx[1][i]]
                    u_s[i] = cropped_UB[idx[0][i], idx[1][i]]
                    if l_s[i] == u_s[i]:
                        gamma = l_s[i]

                if gamma == np.inf:
                    gamma = (np.sum(u_s / (u_s - l_s)) - 1) / np.sum(1 / (u_s - l_s))
                    if gamma < np.max(l_s):
                        gamma = np.max(l_s)
                    elif gamma > np.min(u_s):
                        gamma = np.min(u_s)
                    weights = ((u_s - gamma) / (u_s - l_s)).astype(np.float32)
                else:
                    weights = np.zeros(len(idx[0]), dtype=np.float32)
                    w_partial_sum = 0
                    num_equal = 0
                    for i in range(len(idx[0])):
                        if l_s[i] != u_s[i]:
                            weights[i] = (u_s[i] - gamma) / (u_s[i] - l_s[i])
                            w_partial_sum += weights[i]
                        else:
                            num_equal += 1
                    gap = (1 - w_partial_sum) / num_equal
                    if gap < 0.0:
                        gap = 0.0
                    elif gap > 1.0:
                        gap = 1.0
                    for i in range(len(idx[0])):
                        if l_s[i] == u_s[i]:
                            weights[i] = gap

                for i in range(len(idx[0])):
                    t = idx[0][i]
                    u = idx[1][i]
                    alpha_u[t, u, x, y, r] = weights[i]
                    alpha_l[t, u, x, y, r] = weights[i]
                beta_u[x, y, r] = gamma - np.dot(weights, l_s)
                growth_rate = np.sum(weights)
                if growth_rate <= 1:
                    beta_l[x, y, r] = np.min(l_s) * (1 - growth_rate)
                else:
                    beta_l[x, y, r] = np.max(u_s) * (1 - growth_rate)
    return alpha_u, alpha_l, beta_u, beta_l


@njit
def upper_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0] * (A.shape[3] - 1) + pool_size[0],
                      inner_stride[1] * (A.shape[4] - 1) + pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t + stride[0] * inner_stride[0] * x - inner_stride[0] * pad[0] - inner_pad[0]
                    inner_index_y = u + stride[1] * inner_stride[1] * x - inner_stride[1] * pad[2] - inner_pad[2]
                    if 0 <= inner_index_x < inner_shape[0] and 0 <= inner_index_y < inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0 <= t - inner_stride[0] * p < alpha_u.shape[0] and 0 <= u - inner_stride[1] * q < \
                                        alpha_u.shape[1] and 0 <= p + stride[0] * x - pad[0] < alpha_u.shape[
                                    2] and 0 <= q + stride[1] * y - pad[2] < alpha_u.shape[3]:
                                    A_new[x, y, :, t, u, :] += A_plus[x, y, :, p, q, :] * alpha_u[
                                                                                          t - inner_stride[0] * p,
                                                                                          u - inner_stride[1] * q,
                                                                                          p + stride[0] * x - pad[0],
                                                                                          q + stride[1] * y - pad[2], :]
                                    A_new[x, y, :, t, u, :] += A_minus[x, y, :, p, q, :] * alpha_l[
                                                                                           t - inner_stride[0] * p,
                                                                                           u - inner_stride[1] * q,
                                                                                           p + stride[0] * x - pad[0],
                                                                                           q + stride[1] * y - pad[2],
                                                                                           :]
    B_new = conv_full(A_plus, beta_u, pad, stride) + conv_full(A_minus, beta_l, pad, stride) + B
    return A_new, B_new


@njit
def lower_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0] * (A.shape[3] - 1) + pool_size[0],
                      inner_stride[1] * (A.shape[4] - 1) + pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t + stride[0] * inner_stride[0] * x - inner_stride[0] * pad[0] - inner_pad[0]
                    inner_index_y = u + stride[1] * inner_stride[1] * x - inner_stride[1] * pad[2] - inner_pad[2]
                    if 0 <= inner_index_x < inner_shape[0] and 0 <= inner_index_y < inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0 <= t - inner_stride[0] * p < alpha_u.shape[0] and 0 <= u - inner_stride[1] * q < \
                                        alpha_u.shape[1] and 0 <= p + stride[0] * x - pad[0] < alpha_u.shape[
                                    2] and 0 <= q + stride[1] * y - pad[2] < alpha_u.shape[3]:
                                    A_new[x, y, :, t, u, :] += A_plus[x, y, :, p, q, :] * alpha_l[
                                                                                          t - inner_stride[0] * p,
                                                                                          u - inner_stride[1] * q,
                                                                                          p + stride[0] * x - pad[0],
                                                                                          q + stride[1] * y - pad[2], :]
                                    A_new[x, y, :, t, u, :] += A_minus[x, y, :, p, q, :] * alpha_u[
                                                                                           t - inner_stride[0] * p,
                                                                                           u - inner_stride[1] * q,
                                                                                           p + stride[0] * x - pad[0],
                                                                                           q + stride[1] * y - pad[2],
                                                                                           :]
    B_new = conv_full(A_plus, beta_l, pad, stride) + conv_full(A_minus, beta_u, pad, stride) + B
    return A_new, B_new


# Main function to find bounds at each layer
def compute_bounds(weights, biases, out_shape, nlayer, x0, L, U, p_n, strides, pads, sizes, LBs, UBs):
    pad = (0, 0, 0, 0)
    stride = (1, 1)
    modified_LBs = LBs + (np.ones(out_shape, dtype=np.float32),)
    modified_UBs = UBs + (np.ones(out_shape, dtype=np.float32),)
    resnet_counter = 0
    resnet_last = None
    for i in range(nlayer - 1, -1, -1):
        if resnet_counter > 0:
            resnet_counter -= 1
        if not np.isnan(weights[i]).any() and sizes[i] is None:  # Conv
            if i == nlayer - 1:
                A_u = weights[i].reshape((1, 1, weights[i].shape[0], weights[i].shape[1], weights[i].shape[2],
                                          weights[i].shape[3])) * np.ones((out_shape[0], out_shape[1],
                                                                           weights[i].shape[0], weights[i].shape[1],
                                                                           weights[i].shape[2], weights[i].shape[3]),
                                                                          dtype=np.float32)
                B_u = biases[i] * np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i],
                                            modified_UBs[i].shape, modified_LBs[i + 1], modified_UBs[i + 1])
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i],
                                            modified_LBs[i].shape, modified_LBs[i + 1], modified_UBs[i + 1])
            if resnet_counter == 1:
                A1_l, A1_u = resnet_last
                height_diff = A_l.shape[3] - A1_l.shape[3]
                width_diff = A_l.shape[4] - A1_l.shape[4]
                assert height_diff % 2 == 0
                assert width_diff % 2 == 0
                d_h = height_diff // 2
                d_w = width_diff // 2
                A_l[:, :, :, d_h:A_l.shape[3] - d_h, d_w:A_l.shape[4] - d_w, :] += A1_l
                A_u[:, :, :, d_h:A_u.shape[3] - d_h, d_w:A_u.shape[4] - d_w, :] += A1_u
                resnet_counter = 0
        elif sizes[i] == 'res':  # Resnet
            if i == nlayer - 1:
                A_u = np.eye(out_shape[2]).astype(np.float32).reshape(
                    (1, 1, out_shape[2], 1, 1, out_shape[2])) * np.ones(
                    (out_shape[0], out_shape[1], out_shape[2], 1, 1, out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                resnet_last = (A_l, A_u)
                A_u = weights[i].reshape((1, 1, weights[i].shape[0], weights[i].shape[1], weights[i].shape[2],
                                          weights[i].shape[3])) * np.ones((out_shape[0], out_shape[1],
                                                                           weights[i].shape[0], weights[i].shape[1],
                                                                           weights[i].shape[2], weights[i].shape[3]),
                                                                          dtype=np.float32)
                B_u = biases[i] * np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                resnet_last = (A_l, A_u)
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i],
                                            modified_UBs[i].shape, modified_LBs[i + 1], modified_UBs[i + 1])
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i],
                                            modified_LBs[i].shape, modified_LBs[i + 1], modified_UBs[i + 1])
            resnet_counter += 2
        else:  # Pool
            if i == nlayer - 1:
                A_u = np.eye(out_shape[2]).astype(np.float32).reshape(
                    (1, 1, out_shape[2], 1, 1, out_shape[2])) * np.ones(
                    (out_shape[0], out_shape[1], out_shape[2], 1, 1, out_shape[2]), dtype=np.float32)
                B_u = np.zeros(out_shape, dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            A_u, B_u = upper_bound_pool(A_u, B_u, pad, stride, weights[i].shape[1:], pads[i], strides[i],
                                        modified_UBs[i].shape, np.maximum(modified_LBs[i], 0),
                                        np.maximum(modified_UBs[i], 0))
            A_l, B_l = lower_bound_pool(A_l, B_l, pad, stride, weights[i].shape[1:], pads[i], strides[i],
                                        modified_LBs[i].shape, np.maximum(modified_LBs[i], 0),
                                        np.maximum(modified_UBs[i], 0))
        pad = (
            strides[i][0] * pad[0] + pads[i][0], strides[i][0] * pad[1] + pads[i][1],
            strides[i][1] * pad[2] + pads[i][2],
            strides[i][1] * pad[3] + pads[i][3])
        stride = (strides[i][0] * stride[0], strides[i][1] * stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, L, U, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, L, U, p_n)
    return LLB, ULB, LUB, UUB


# Main function to find output bounds
def find_output_bounds(weights, biases, shapes, pads, strides, sizes, x0, L, U, p_n):
    LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, L, U, p_n)
    LBs = [L, LB]
    UBs = [U, UB]
    for i in range(2, len(weights) + 1):
        LB, _, _, UB = compute_bounds(tuple(weights), tuple(biases), shapes[i], i, x0, L, U, p_n, tuple(strides),
                                      tuple(pads), tuple(sizes), tuple(LBs), tuple(UBs))
        UBs.append(UB)
        LBs.append(LB)
    return LBs[-1], UBs[-1]


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
    fn(weights, biases, shapes, model.pads, model.strides, x, -eps_0 * np.ones_like(x), eps_0 * np.ones_like(x), p_n)


# Runs CNN with input x
def run_inp(weights, biases, shapes, pads, strides, sizes, x, act):
    layers = []
    for i in range(len(weights)):
        if i != 0:
            x = act(x)
        if not np.isnan(weights[i]).any() and sizes[i] is None:  # Conv
            x = conv(weights[i], x, np.asarray(pads[i]), np.asarray(strides[i])) + biases[i]
            layers.append(x)
        elif sizes[i] == 'res':  # Resnet
            x = conv(weights[i], x, np.asarray(pads[i]), np.asarray(strides[i])) + biases[i] + layers[-2]
            layers.append(x)
        else:
            x = pool(np.asarray(sizes[i]), x, np.asarray(pads[i]), np.asarray(strides[i]))
            layers.append(x)
    return x


import csv
import pickle


# Certify with linear bounding certifier
def certify(network, strides, paddings, epss, n_pts=1000, test=True, cifar=False, normalize=False, act='relu',
            bnd_fn=relu_linear_bounds, norm=105):
    if cifar:
        data = CIFAR()
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

    global linear_bounds
    if act == 'relu':
        act_fn = lambda x: np.maximum(x, 0)
    elif act == 'tanh':
        act_fn = np.tanh
    linear_bounds = bnd_fn
    print('Recompiling')
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()

    init = None
    for t in tests:
        init = t[0]
        break

    results = []
    correct = 0
    for test in tests:
        image = test[0]
        true_label = test[1]
        out = run_inp(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes,
                      image.astype(np.float32), act_fn)
        predict_label = np.argmax(np.squeeze(out))
        if int(predict_label) == int(true_label):
            correct += 1
    results.append(correct / len(tests))

    start_time = time.time()
    print("Network = {}".format(network))
    eps_results = [0 for i in epss]
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

                correct += 1
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
                        LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides, model.sizes,
                                                    image.astype(np.float32), L.astype(np.float32),
                                                    U.astype(np.float32), norm)
                        # print("[L3] LB = {}, UB = {}, t = {}".format(LB, UB, t))
                        if LB < 0:
                            failed = True
                            break
                if not failed:
                    eps_results[i] += 1
                else:
                    break
    results = results + [correct / len(tests) for correct in eps_results]
    return results


# Certify with combination of linear bounding certifier and IBP
def certify_fastlinibp(network, strides, paddings, epss, n_pts=1000, test=True, cifar=False, normalize=False,
                       act='relu',
                       bnd_fn=relu_linear_bounds, norm=105):
    if cifar:
        data = CIFAR()
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
    idx = np.random.permutation(np.arange(x_val.shape[0]))[:n_pts]
    x_val = x_val[idx, :, :, :]
    y_val = y_val[idx, :]
    vals = []
    for i in range(n_pts):
        vals.append((np.float32(x_val[i, :, :, :]), int(np.argmax(y_val[i, :]))))
    tests = vals

    final = []
    print('Loading model')
    with open('networks/' + network + '.pkl', 'rb') as file:
        param_vals = pickle.load(file)
    if cifar:
        model = CNNModel(param_vals, strides, paddings, inp_shape=(32, 32, 3))
    else:
        model = CNNModel(param_vals, strides, paddings, inp_shape=(28, 28, 1))

    global linear_bounds
    if act == 'relu':
        act_fn = lambda x: np.maximum(x, 0)
    elif act == 'tanh':
        act_fn = np.tanh
    linear_bounds = bnd_fn
    print('Recompiling')
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()

    init = None
    for t in tests:
        init = t[0]
        break

    results = []
    correct = 0
    for test in tests:
        image = test[0]
        true_label = test[1]
        out = run_inp(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes,
                      image.astype(np.float32), act_fn)
        predict_label = np.argmax(np.squeeze(out))
        if int(predict_label) == int(true_label):
            correct += 1
    results.append(correct / len(tests))

    print("Network = {}".format(network))
    eps_results = [0 for i in epss]
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

                correct += 1
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
                        LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides, model.sizes,
                                                    image.astype(np.float32), L.astype(np.float32),
                                                    U.astype(np.float32), norm)
                        LB_ibp, UB_ibp = find_output_bounds_ibp(weights, biases, shapes, model.pads, model.strides,
                                                                image.astype(np.float32), act_fn,
                                                                L.astype(np.float32), U.astype(np.float32))

                        # Pick better bound
                        LB = np.maximum(LB, LB_ibp)

                        if LB < 0:
                            failed = True
                            break
                if not failed:
                    eps_results[i] += 1
                else:
                    break
    results = results + [correct / len(tests) for correct in eps_results]
    return results


if __name__ == '__main__':
    networks = ['ibp_mnist_ada_002',
                'mnist_small_singleprop_cnncertzero_ada_lr_0005_3_100']

    final = []
    for n in networks:
        results = certify(n, [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                          [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45], bnd_fn=zero_linear_bounds)
        results = [str(v) for v in results]
        print('\t'.join(results))
        final.append('\t'.join(results))
    for f in final:
        print(f)
    print('MNIST small Zero')

    final = []
    for n in networks:
        results = certify_fastlinibp(n, [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                                     [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                     bnd_fn=zero_linear_bounds)
        results = [str(v) for v in results]
        print('\t'.join(results))
        final.append('\t'.join(results))
    for f in final:
        print(f)
    print('MNIST small Zero+IBP')

    networks = ['ibp_cifar_ada_0005',
                'cifar_small_singleprop_fastlin_ada_lr_0005_8255_350']
    final = []
    for n in networks:
        results = certify(n, [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                          [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255,
                           10 / 255],
                          cifar=True, normalize=True, bnd_fn=relu_linear_bounds)
        results = [str(v) for v in results]
        print('\t'.join(results))
        final.append('\t'.join(results))
    for f in final:
        print(f)
    print('CIFAR small Fast-Lin')

    final = []
    for n in networks:
        results = certify_fastlinibp(n, [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                                     [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255,
                                      10 / 255],
                                     cifar=True, normalize=True, bnd_fn=relu_linear_bounds)
        results = [str(v) for v in results]
        print('\t'.join(results))
        final.append('\t'.join(results))
    for f in final:
        print(f)
    print('CIFAR small Fast-Lin+IBP')


