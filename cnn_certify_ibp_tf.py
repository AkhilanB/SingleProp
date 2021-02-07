"""
cnn_certify_ibp_tf.py

Certifies networks under IBP certification

Copyright (C) 2021, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Sijia Liu <liusiji5@msu.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""
import numpy as np
from setup_mnist import MNIST
from setup_cifar import CIFAR
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from load_model import load_model
import random
import time

part = 1

import sys

if len(sys.argv) > 1:
    part = int(sys.argv[1])


# Certifies with IBP
def certify(network, sess, filters, kernels, strides, paddings, epss, n_pts=100, test=True, cifar=False,
            normalize=False, batch_size=100):
    tf.set_random_seed(99)
    random.seed(99)

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
    if n_pts is None:
        n_pts = x_val.shape[0]  # Full test set
    idx = np.random.permutation(np.arange(x_val.shape[0]))[:n_pts]
    x_val = x_val[idx, :, :, :]
    y_val = y_val[idx, :]
    vals = []
    for i in range(n_pts):
        vals.append((np.float32(x_val[i, :, :, :]), int(np.argmax(y_val[i, :]))))
    tests = vals

    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
    model = load_model(network, sess, filters, kernels, strides, paddings)
    eps = tf.placeholder('float', shape=())

    x0 = inputs

    if normalize:
        U = x0 + eps / np.asarray([0.2023, 0.1994, 0.2010])
        L = x0 - eps / np.asarray([0.2023, 0.1994, 0.2010])
        U = tf.clip_by_value(U, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
        L = tf.clip_by_value(L, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
    else:
        U = tf.clip_by_value(x0 + eps, 0, 1)
        L = tf.clip_by_value(x0 - eps, 0, 1)

    lb, ub = model.ibp(L, U)

    np.random.seed(99)

    epss = [0] + epss
    start_time = time.time()
    print("Network = {}".format(network))

    results = []
    for eps_val in epss:
        success = 0
        for batch in range(x_val.shape[0] // batch_size):
            feed_dict = {inputs: x_val[batch_size * batch:batch_size * (batch + 1)], eps: eps_val}
            lb_val, ub_val = sess.run([lb, ub], feed_dict=feed_dict)

            for i in range(batch_size):
                true_label = tests[i + batch_size * batch][1]
                failed = False
                for k in range(10):
                    if lb_val[true_label][k][i] < 0:
                        failed = True
                        break
                if not failed:
                    success += 1
        results.append(success / n_pts)
    print('Time = {}'.format(str(time.time() - start_time)))
    return results


# Finds approximation error metrics
def metrics(network, sess, filters, kernels, strides, paddings, epss, n_pts=100, test=True, cifar=False,
            normalize=False, batch_size=100):
    tf.set_random_seed(99)
    random.seed(99)

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
    if n_pts is None:
        n_pts = x_val.shape[0]  # Full test set
    idx = np.random.permutation(np.arange(x_val.shape[0]))[:n_pts]
    x_val = x_val[idx, :, :, :]
    y_val = y_val[idx, :]
    vals = []
    for i in range(n_pts):
        vals.append((np.float32(x_val[i, :, :, :]), int(np.argmax(y_val[i, :]))))

    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
    model = load_model(network, sess, filters, kernels, strides, paddings)
    eps = tf.placeholder('float', shape=())

    x0 = inputs

    if normalize:
        U = x0 + eps / np.asarray([0.2023, 0.1994, 0.2010])
        L = x0 - eps / np.asarray([0.2023, 0.1994, 0.2010])
        U = tf.clip_by_value(U, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
        L = tf.clip_by_value(L, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
    else:
        U = tf.clip_by_value(x0 + eps, 0, 1)
        L = tf.clip_by_value(x0 - eps, 0, 1)

    ibp_layers = model.ibp(L, U, all_layers=True)
    layers = model.predict(x0, all_layers=True)

    np.random.seed(99)

    epss = [0] + epss
    print("Network = {}".format(network))

    eps_val = epss[0]
    full_error1 = []
    full_error2 = []
    for batch in range(x_val.shape[0] // batch_size):
        feed_dict = {inputs: x_val[batch_size * batch:batch_size * (batch + 1)], eps: eps_val}
        ibp_layer_vals, layer_vals = sess.run([ibp_layers, layers], feed_dict=feed_dict)

        error1 = None
        error2 = None
        for ibp_layer_val, layer_val in zip(ibp_layer_vals, layer_vals):
            L_layer_val, U_layer_val = ibp_layer_val
            if error1 is None:
                error1 = np.mean(
                    np.reshape(np.abs(layer_val - 0.5 * (L_layer_val + U_layer_val)), (layer_val.shape[0], -1)),
                    axis=1)
                error2 = np.mean(
                    np.reshape(np.abs(layer_val - 0.5 * (L_layer_val + U_layer_val)) / (
                            U_layer_val - L_layer_val + 0.000001) * np.heaviside(U_layer_val - L_layer_val - 0.000001,
                                                                                 0), (layer_val.shape[0], -1)),
                    axis=1)  # Zero if no bound gap
            else:
                error1 += np.mean(
                    np.reshape(np.abs(layer_val - 0.5 * (L_layer_val + U_layer_val)), (layer_val.shape[0], -1)),
                    axis=1)
                error2 = +np.mean(
                    np.reshape(np.abs(layer_val - 0.5 * (L_layer_val + U_layer_val)) / (
                            U_layer_val - L_layer_val + 0.000001) * np.heaviside(U_layer_val - L_layer_val - 0.000001,
                                                                                 0), (layer_val.shape[0], -1)),
                    axis=1)  # Zero if no bound gap
        full_error1.append(error1)
        full_error2.append(error2)
    full_error1 = np.concatenate(full_error1)
    full_error2 = np.concatenate(full_error2)
    return np.mean(full_error1), np.std(full_error1), np.mean(full_error2), np.std(full_error2)


# Combines IBP model certifications of multiple networks
def certify_combined(networks, sess, filters, kernels, strides, paddings, epss, n_pts=100, test=True, cifar=False,
                     normalize=False, batch_size=100, filter=False):
    tf.set_random_seed(99)
    random.seed(99)

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
    if n_pts is None:
        n_pts = x_val.shape[0]  # Full test set
    idx = np.random.permutation(np.arange(x_val.shape[0]))[:n_pts]
    x_val = x_val[idx, :, :, :]
    y_val = y_val[idx, :]
    vals = []
    for i in range(n_pts):
        vals.append((np.float32(x_val[i, :, :, :]), int(np.argmax(y_val[i, :]))))
    tests = vals

    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
    models = []
    for network in networks:
        model = load_model(network, sess, filters, kernels, strides, paddings)
        models.append(model)
    eps = tf.placeholder('float', shape=())

    x0 = inputs

    if normalize:
        U = x0 + eps / np.asarray([0.2023, 0.1994, 0.2010])
        L = x0 - eps / np.asarray([0.2023, 0.1994, 0.2010])
        U = tf.clip_by_value(U, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
        L = tf.clip_by_value(L, -np.asarray([0.4914, 0.4822, 0.4465]) / np.asarray([0.2023, 0.1994, 0.2010]),
                             (1 - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010]))
    else:
        U = tf.clip_by_value(x0 + eps, 0, 1)
        L = tf.clip_by_value(x0 - eps, 0, 1)

    lbs = []
    ubs = []
    for model in models:
        lb, ub = model.ibp(L, U)
        lbs.append(lb)
        ubs.append(ub)

    if filter:
        outs = []
        for model in models:
            out = model.predict(x0)
            outs.append(out)

    np.random.seed(99)

    epss = [0] + epss
    start_time = time.time()

    results = []
    for eps_val in epss:
        success = 0
        for batch in range(x_val.shape[0] // batch_size):
            feed_dict = {inputs: x_val[batch_size * batch:batch_size * (batch + 1)], eps: eps_val}
            lb_vals, ub_vals = sess.run([lbs, ubs], feed_dict=feed_dict)
            if filter:
                out_vals = sess.run(outs, feed_dict=feed_dict)

            for i in range(batch_size):
                verified = False
                for lb_val in lb_vals:
                    true_label = tests[i + batch_size * batch][1]
                    failed = False
                    for k in range(10):
                        if lb_val[true_label][k][i] < 0:
                            failed = True
                            break
                    if not failed:
                        verified = True
                        success += 1
                        break
                if filter and verified:
                    for out_val in out_vals:
                        true_label = tests[i + batch_size * batch][1]
                        failed = False
                        for k in range(10):
                            if out_val[i, true_label] < out_val[i, k]:
                                failed = True
                                break
                        if failed:
                            success -= 1
                            break

        results.append(success / n_pts)
    print('Time = {}'.format(str(time.time() - start_time)))
    return results


if __name__ == '__main__':
    final = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if part == 1:  # MNIST Small
            networks = ['ibp_mnist_001',
                        'ibp_mnist_ada_002',
                        'ibp_mnist_ada_002_v2',
                        'ibp_mnist_ada_002_v3',
                        'ibp_mnist_ada_002_v4',
                        'ibp_mnist_ada_002_v5',
                        'mnist_small_singleprop_cnncertzero_lr_0005_3_100',
                        'mnist_small_singleprop_cnncertzero_ada_lr_0005_3_100',
                        'mnist_small_singleprop_seed_101_cnncertzero_ada_lr_0005_3_150',
                        'mnist_small_singleprop_seed_102_cnncertzero_ada_lr_0005_3_150',
                        'mnist_small_singleprop_seed_103_cnncertzero_ada_lr_0005_3_100',
                        'mnist_small_singleprop_seed_104_cnncertzero_ada_lr_0005_3_100',
                        'mnist_small_normal_100',
                        'mnist_small_adv_3_100',
                        'mnist_small_trades_3_100']

            final = []
            for n in networks:
                results = certify(n, sess, [16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                  ['SAME', 'SAME', 'VALID', 'SAME'], [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                  n_pts=None)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST small')
        elif part == 2:  # CIFAR Small
            networks = ['ibp_cifar_001',
                        'ibp_cifar_ada_0005',
                        'ibp_cifar_ada_0005_v2',
                        'ibp_cifar_ada_0005_v3',
                        'cifar_small_singleprop_fastlin_ada_lr_001_8255_350',
                        'cifar_small_singleprop_fastlin_ada_lr_0005_8255_350'
                        'cifar_small_singleprop_seed_101_fastlin_ada_lr_0005_8255_350',
                        'cifar_small_singleprop_seed_102_fastlin_ada_lr_0005_8255_350']

            final = []
            for n in networks:
                results = certify(n, sess, [16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                  ['SAME', 'SAME', 'VALID', 'SAME'],
                                  [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255, 10 / 255],
                                  cifar=True, normalize=True, n_pts=None)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('CIFAR small')
        elif part == 3:  # MNIST Medium
            networks = ['ibp_medium_mnist_0002',
                        'ibp_medium_mnist_ada_0002',
                        'mnist_medium_singleprop_cnncertzero_lr_001_3_100',
                        'mnist_medium_singleprop_cnncertzero_ada_lr_001_3_100']

            filters = [32, 32, 64, 64, 512, 512, 10]
            kernels = [3, 4, 3, 4, 4, 1, 1]
            strides = [1, 2, 1, 2, 1, 1, 1]
            paddings = ['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID']

            final = []
            for n in networks:
                results = certify(n, sess, filters, kernels, strides, paddings,
                                  [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                  n_pts=100)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST medium')
        elif part == 4:  # MNIST Wide
            filters = [128, 256, 512, 1024, 10]
            kernels = [3, 3, 3, 7, 1]
            strides = [1, 2, 2, 1, 1]
            paddings = ['SAME', 'SAME', 'SAME', 'VALID', 'SAME']

            networks = ['ibp_wide_mnist_001',
                        'mnist_wide_singleprop_cnncertzero_lr_001_3_100',
                        'mnist_wide_adv_lr_001_3_100',
                        'mnist_wide_normal_lr_001']

            final = []
            for n in networks:
                results = certify(n, sess, filters, kernels, strides, paddings,
                                  [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                  n_pts=None)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST wide')
        elif part == 5:  # CIFAR Large
            networks = ['cifar_large_singlemargin_fastlin_lr_0001_8255_350',
                        'ibp_large_cifar_0005']
            filters = [64, 64, 128, 128, 128, 512, 10]
            kernels = [3, 3, 3, 3, 3, 16, 1]
            strides = [1, 1, 2, 1, 1, 1, 1]
            paddings = ['SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'VALID', 'SAME']
            final = []
            for n in networks:
                results = certify(n, sess, filters, kernels, strides, paddings,
                                  [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255, 10 / 255],
                                  cifar=True, normalize=True, n_pts=None)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('CIFAR large')
        elif part == 6:  # Combined model accuracies
            networks = ['ibp_mnist_ada_002',
                        'mnist_small_singleprop_cnncertzero_ada_lr_0005_3_100']
            final = []
            results = certify_combined(networks, sess, [16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                       ['SAME', 'SAME', 'VALID', 'SAME'],
                                       [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                       n_pts=None)
            results = [str(v) for v in results]
            print('\t'.join(results))
            final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST small combined')

            networks = ['ibp_cifar_ada_0005',
                        'cifar_small_singleprop_fastlin_ada_lr_0005_8255_350']

            final = []
            results = certify_combined(networks, sess, [16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                       ['SAME', 'SAME', 'VALID', 'SAME'],
                                       [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255,
                                        10 / 255],
                                       cifar=True, normalize=True, n_pts=None, filter=True)
            results = [str(v) for v in results]
            print('\t'.join(results))
            final.append('\t'.join(results))
            for f in final:
                print(f)
            print('CIFAR small combined')

            networks = ['ibp_medium_mnist_ada_0002',
                        'mnist_medium_singleprop_cnncertzero_ada_lr_001_3_100']
            filters = [32, 32, 64, 64, 512, 512, 10]
            kernels = [3, 4, 3, 4, 4, 1, 1]
            strides = [1, 2, 1, 2, 1, 1, 1]
            paddings = ['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID']
            final = []
            results = certify_combined(networks, sess, filters, kernels, strides, paddings,
                                       [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.45],
                                       n_pts=None)
            results = [str(v) for v in results]
            print('\t'.join(results))
            final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST medium combined')

            networks = ['cifar_large_singlemargin_fastlin_lr_0001_8255_350',
                        'ibp_large_cifar_0005']
            filters = [64, 64, 128, 128, 128, 512, 10]
            kernels = [3, 3, 3, 3, 3, 16, 1]
            strides = [1, 1, 2, 1, 1, 1, 1]
            paddings = ['SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'VALID', 'SAME']
            final = []
            results = certify_combined(networks, sess, filters, kernels, strides, paddings,
                                       [0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 5 / 255, 7 / 255, 8 / 255, 9 / 255,
                                        10 / 255],
                                       cifar=True, normalize=True, n_pts=None)
            results = [str(v) for v in results]
            print('\t'.join(results))
            final.append('\t'.join(results))
            for f in final:
                print(f)
            print('CIFAR large combined')
        elif part == 7:  # Approximation error metrics
            networks = ['ibp_mnist_ada_002',
                        'mnist_small_singlemargin_cnncertzero_ada_lr_0005_3_100']

            final = []
            for n in networks:
                results = metrics(n, sess, [16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                  ['SAME', 'SAME', 'VALID', 'SAME'], [0.3],
                                  n_pts=None)
                results = [str(v) for v in results]
                print('\t'.join(results))
                final.append('\t'.join(results))
            for f in final:
                print(f)
            print('MNIST small')
