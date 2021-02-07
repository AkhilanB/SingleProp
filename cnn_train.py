"""
cnn_train.py

Trains networks with SingleProp and other baselines

Copyright (C) 2021, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Sijia Liu <liusiji5@msu.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from setup_mnist import MNIST
from setup_cifar import CIFAR
import numpy as np
import pickle
import time

part = 1

import sys

if len(sys.argv) > 1 and __name__ == '__main__':
    part = int(sys.argv[1])

# Disable warnings
tf.logging.set_verbosity(tf.logging.ERROR)


def augment(data):
    a = np.random.randint(0, 9, data.shape[0])
    b = np.random.randint(0, 9, data.shape[0])
    flip = np.random.randint(2, size=data.shape[0])
    new_x = []
    for i in range(data.shape[0]):
        x = data[i, :, :, :]
        x = np.pad(x, ((4, 4), (4, 4), (0, 0)), 'constant')
        if flip[i] == 1:
            x = np.fliplr(x)
        x = x[a[i]:a[i] + 32, b[i]:b[i] + 32, :]
        new_x.append(x)
    new_data = np.stack(new_x)
    return new_data


def save(data, name):
    with open('networks/' + str(name) + '.pkl', 'wb') as file:
        pickle.dump(data, file)


def train_trades(filters, kernels, strides, paddings, name, eps_val, lr_val, adv_steps=40, step_size=0.01,
                 batch_size=100, EPOCHS=25,
                 cifar=False, act=tf.nn.relu, start=None):
    if cifar:
        data = CIFAR()
    else:
        data = MNIST()
    x_train = data.train_data + 0.5
    y_train = data.train_labels

    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
        last_shape = 3
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
        last_shape = 1

    if start is not None:
        with open('networks/' + start + '.pkl', 'rb') as file:
            param_vals = pickle.load(file)

    x0 = inputs
    params = []
    x = x0
    x_adv = inputs_adv
    layers = [x]
    layers_adv = [x_adv]
    np.random.seed(99)
    # Define base network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        if l == 'pool':
            x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            x_adv = tf.nn.max_pool(x_adv, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
            W = tf.fill([k, k], np.nan)
            b = tf.fill([], np.nan)
            params.append((W, b))
            layers.append(x)
            layers_adv.append(x_adv)
        elif type(s) is str:  # Residual
            s = int(s[1:])
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b + layers[-2]
            layers.append(x)
            x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b + layers[-2]
            layers_adv.append(x_adv)
        else:  # Conv
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            if start is not None:
                W_val = param_vals[i][0]
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            if start is not None:
                b_val = param_vals[i][1]
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            if i == 0:
                x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(x_adv, W, [1, s, s, 1], p) + b
            else:
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b
            layers.append(x)
            layers_adv.append(x_adv)
    logits = tf.layers.flatten(x)
    logits_adv = tf.layers.flatten(x_adv)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)

    trades_loss = tf.einsum('ij,ij->i', probs, log_probs - tf.nn.log_softmax(logits_adv))
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    cross_entropy = normal_loss + trades_loss

    # Code for attack
    grad = tf.gradients(cross_entropy, inputs)[0]

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                x_adv = x_train[idx, :, :, :].copy()
                x_nat = x_train[idx, :, :, :].copy()
                for j in range(adv_steps):
                    feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :]}
                    grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                    delta = step_size * np.sign(grad_val)
                    x_adv = x_adv + delta
                    x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                    x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 10 == 0:
                if start:
                    start_epochs = str(start.rsplit('_', 1)[1])
                else:
                    start_epochs = 0
                save(sess.run(params), name.rsplit('_', 1)[0] + '_' + str(start_epochs + epoch + 1))
        save(sess.run(params), name)
    tf.reset_default_graph()
    return str(time.time() - start_time)


def train_adv(filters, kernels, strides, paddings, name, eps_val, lr_val, adv_steps=40, step_size=0.01, batch_size=100,
              EPOCHS=25,
              cifar=False, act=tf.nn.relu, start=None):
    if cifar:
        data = CIFAR()
    else:
        data = MNIST()
    x_train = data.train_data + 0.5
    y_train = data.train_labels

    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        last_shape = 3
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        last_shape = 1

    if start is not None:
        with open('networks/' + start + '.pkl', 'rb') as file:
            param_vals = pickle.load(file)

    x0 = inputs
    params = []
    x = x0
    layers = [x]
    np.random.seed(99)
    # Define base network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        if l == 'pool':
            x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            W = tf.fill([k, k], np.nan)
            b = tf.fill([], np.nan)
            params.append((W, b))
            layers.append(x)
        elif type(s) is str:  # Residual
            s = int(s[1:])
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b + layers[-2]
            layers.append(x)
        else:  # Conv
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            if start is not None:
                W_val = param_vals[i][0]
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            if start is not None:
                b_val = param_vals[i][1]
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            if i == 0:
                x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
            else:
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
            layers.append(x)
    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # Code for attack
    grad = tf.gradients(cross_entropy, inputs)[0]

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    start_time = time.time()
    # print('Time ' + str(time.time()-start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                x_adv = x_train[idx, :, :, :]
                x_nat = x_train[idx, :, :, :]
                for j in range(adv_steps):
                    feed_dict_attack = {inputs: x_adv, labels: y_train[idx, :]}
                    grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                    delta = step_size * np.sign(grad_val)
                    x_adv = x_adv + delta
                    x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                    x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 10 == 0:
                if start:
                    start_epochs = str(start.rsplit('_', 1)[1])
                else:
                    start_epochs = 0
                save(sess.run(params), name.rsplit('_', 1)[0] + '_' + str(start_epochs + epoch + 1))
        save(sess.run(params), name)
    tf.reset_default_graph()
    return str(time.time() - start_time)


def train_normal(filters, kernels, strides, paddings, name, lr_val, l1=False, batch_size=100, EPOCHS=25, cifar=False,
                 act=tf.nn.relu):
    if cifar:
        data = CIFAR()
    else:
        data = MNIST()
    x_train = data.train_data + 0.5
    y_train = data.train_labels
    x_test = data.validation_data + 0.5
    y_test = data.validation_labels

    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        last_shape = 3
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        last_shape = 1

    x0 = inputs
    params = []
    x = x0
    layers = [x]
    weight_reg = 0
    np.random.seed(99)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        if l == 'pool':
            x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            W = tf.fill([k, k], np.nan)
            b = tf.fill([], np.nan)
            params.append((W, b))
            layers.append(x)
        elif type(s) is str:  # Residual
            s = int(s[1:])
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            W_flat = tf.reshape(W, (k * k * last_shape, l))
            weight_reg += tf.norm(tf.transpose(W_flat), ord=np.inf, axis=(0, 1))

            params.append((W, b))
            last_shape = l
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b + layers[-2]
            layers.append(x)
        else:  # Conv
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            W_flat = tf.reshape(W, (k * k * last_shape, l))
            weight_reg += tf.norm(tf.transpose(W_flat), ord=np.inf, axis=(0, 1))

            params.append((W, b))
            last_shape = l
            if i == 0:
                x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
            else:
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
            layers.append(x)
    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if l1:
        cross_entropy = normal_cross_entropy + 0.00002 * weight_reg
    else:
        cross_entropy = normal_cross_entropy

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :, :, :], labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if epoch % 10 == 0:
                save(sess.run(params), name.rsplit('_', 1)[0] + str(epoch + 1))
                feed_dict_test = {inputs: x_test, labels: y_test}
                accuracy_value = sess.run(accuracy, feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    return str(time.time() - start)


def train_ibp(filters, kernels, strides, paddings, name, eps_val, K_val, lr_val, ada=False, gamma=10,
              batch_size=100, EPOCHS=25, cifar=False, act=tf.nn.relu, seed=99,
              normalize=False):
    if cifar:
        data = CIFAR()
    else:
        data = MNIST()
    x_train = data.train_data + 0.5
    y_train = data.train_labels
    x_test = data.validation_data + 0.5
    y_test = data.validation_labels
    if cifar and normalize:  # normalize
        x_train = (x_train - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])
        x_test = (x_test - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])

    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        last_shape = 3
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        last_shape = 1

    x0 = inputs
    eps = tf.placeholder('float', shape=())
    params = []
    x = x0
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
    np.random.seed(seed)
    layers = [(L, x, U)]
    # Define base network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        if l == 'pool':
            x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            U = tf.nn.max_pool(U, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            L = tf.nn.max_pool(L, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            W = tf.fill([k, k], np.nan)
            b = tf.fill([], np.nan)
            params.append((W, b))
            layers.append((L, x, U))
        elif type(s) is str:  # Residual
            s = int(s[1:])
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            W_plus = tf.nn.relu(W)
            W_minus = -tf.nn.relu(-W)
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b + layers[-2][1]
            U, L = act(U), act(L)
            U, L = tf.nn.conv2d(U, W_plus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_minus, [1, s, s, 1], p) + b + \
                   layers[-2][2], \
                   tf.nn.conv2d(U, W_minus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_plus, [1, s, s, 1], p) + b + \
                   layers[-2][0]
            layers.append((L, x, U))
        else:  # Conv
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            W_plus = tf.nn.relu(W)
            W_minus = -tf.nn.relu(-W)
            b_val = np.zeros((l,))
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            if i == 0:
                x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                U, L = tf.nn.conv2d(U, W_plus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_minus, [1, s, s, 1], p) + b, \
                       tf.nn.conv2d(U, W_minus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_plus, [1, s, s, 1], p) + b
            else:
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                U, L = act(U), act(L)
                U, L = tf.nn.conv2d(U, W_plus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_minus, [1, s, s, 1], p) + b, \
                       tf.nn.conv2d(U, W_minus, [1, s, s, 1], p) + tf.nn.conv2d(L, W_plus, [1, s, s, 1], p) + b
            layers.append((L, x, U))
    logits = tf.layers.flatten(x)
    ub = tf.layers.flatten(U)
    lb = tf.layers.flatten(L)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    adv_logits = labels * lb + (1 - labels) * ub
    adv_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=adv_logits, labels=labels))
    normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    K = tf.placeholder('float', shape=())
    cross_entropy = (1 - K) * adv_cross_entropy + K * normal_cross_entropy

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    start = time.time()
    print('Time ' + str(time.time() - start))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        np.random.seed(99)
        step = 0
        if ada:
            K_val = lambda x: 1
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                if normalize:
                    feed_dict_train = {inputs: augment(x_train[idx, :, :, :]), labels: y_train[idx, :],
                                       eps: eps_val(step), K: K_val(step), lr: lr_val(step)}
                else:
                    feed_dict_train = {inputs: x_train[idx, :, :, :], labels: y_train[idx, :],
                                       eps: eps_val(step), K: K_val(step), lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if ada:
                feed_dict_test = {inputs: x_test, labels: y_test, eps: eps_val(step)}
                l_spec, l_fit = sess.run([adv_cross_entropy, normal_cross_entropy], feed_dict=feed_dict_test)
                K_val = lambda x: l_spec / (l_spec + gamma * l_fit)

        save(sess.run(params), name)
    tf.reset_default_graph()
    return str(time.time() - start)


def get_crown_bounds(z, v):
    negative = tf.less_equal(z + v, tf.zeros_like(v))
    positive = tf.greater_equal(z - v, tf.zeros_like(v))
    intermediate = tf.logical_not(tf.logical_or(positive, negative))
    int_p = tf.logical_and(tf.greater_equal(z, tf.zeros_like(z)), intermediate)
    int_n = tf.logical_and(tf.greater(z, tf.zeros_like(z)), intermediate)

    err = tf.zeros_like(z)
    err = tf.where(positive, v, err)
    err = tf.where(int_p, v, err)
    err = tf.where(int_n, (z + v) / 2, err)
    return err


def get_cnncertzero_bounds(z, v):
    negative = tf.less_equal(z + v, tf.zeros_like(v))
    positive = tf.greater_equal(z - v, tf.zeros_like(v))
    intermediate = tf.logical_not(tf.logical_or(positive, negative))

    err = tf.zeros_like(z)
    err = tf.where(positive, v, err)
    err = tf.where(intermediate, (z + v) / 2, err)
    return err


def get_fastlin_bounds(z, v):
    negative = tf.less_equal(z + v, tf.zeros_like(v))
    positive = tf.greater_equal(z - v, tf.zeros_like(v))
    intermediate = tf.logical_not(tf.logical_or(positive, negative))

    err = tf.zeros_like(z)
    err = tf.where(positive, v, err)
    err = tf.where(intermediate, 3 * v / 4 + z / 2 - z ** 2 / (4 * v + 0.0001), err)
    return err


def train_singleprop(filters, kernels, strides, paddings, name, eps_val, K_val, lr_val, scale=1, ada=False, gamma=10,
                     batch_size=100, EPOCHS=25, normalize=False, cifar=False,
                     cnncertzero=False, crown=False, act=tf.nn.relu, seed=99, init=None):
    if cifar:
        data = CIFAR()
    else:
        data = MNIST()
    x_train = data.train_data + 0.5
    y_train = data.train_labels
    x_test = data.validation_data + 0.5
    y_test = data.validation_labels
    if cifar and normalize:  # normalize
        x_train = (x_train - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])
        x_test = (x_test - np.asarray([0.4914, 0.4822, 0.4465])) / np.asarray([0.2023, 0.1994, 0.2010])

    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        last_shape = 3
    else:
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        last_shape = 1

    x0 = inputs
    eps = tf.placeholder('float', shape=())
    params = []
    x = x0
    np.random.seed(seed)
    # Define base network
    if normalize:
        V = eps * tf.ones_like(inputs) / np.asarray([0.2023, 0.1994, 0.2010])
    else:
        V = eps * tf.ones_like(inputs)

    if init is not None:
        with open('networks/' + init + '.pkl', 'rb') as file:
            param_vals = pickle.load(file)
    else:
        param_vals = None

    layers = [(x, V)]
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        if l == 'pool':
            V = 0.5 * (tf.nn.max_pool(x + V, ksize=[1, k, k, 1],
                                      strides=[1, s, s, 1], padding=p) -
                       tf.nn.max_pool(x - V, ksize=[1, k, k, 1],
                                      strides=[1, s, s, 1], padding=p))

            x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                               strides=[1, s, s, 1], padding=p)
            W = tf.fill([k, k], np.nan)
            b = tf.fill([], np.nan)
            params.append((W, b))
            layers.append((x, V))
        elif type(s) is str:  # Residual
            s = int(s[1:])
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            if param_vals:
                W_val = param_vals[i][0]
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            if param_vals:
                b_val = param_vals[i][1]
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l

            if crown:
                V = get_crown_bounds(x, V)
            elif cnncertzero:
                V = get_cnncertzero_bounds(x, V)
            else:
                V = get_fastlin_bounds(x, V)
            V = scale * tf.nn.conv2d(V, tf.abs(W), [1, s, s, 1], p) + layers[-2][1]
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b + layers[-2][0]
            layers.append((x, V))
        else:  # Conv
            W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
            W_val = W_val.reshape((k, k, last_shape, l))
            if param_vals:
                assert W_val.shape == param_vals[i][0].shape
                W_val = param_vals[i][0]
            W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
            b_val = np.zeros((l,))
            if param_vals:
                assert b_val.shape == param_vals[i][1].shape
                b_val = param_vals[i][1]
            b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

            params.append((W, b))
            last_shape = l
            if i == 0:
                x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                V = scale * tf.nn.conv2d(V, tf.abs(W), [1, s, s, 1], p)
            else:
                if crown:
                    V = get_crown_bounds(x, V)
                elif cnncertzero:
                    V = get_cnncertzero_bounds(x, V)
                else:
                    V = get_fastlin_bounds(x, V)
                V = scale * tf.nn.conv2d(V, tf.abs(W), [1, s, s, 1], p)

                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
            layers.append((x, V))
    logits = tf.layers.flatten(x)

    ub = logits + tf.layers.flatten(V)
    lb = logits - tf.layers.flatten(V)
    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))
    adv_logits = labels * lb + (1 - labels) * ub
    adv_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=adv_logits, labels=labels))
    normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    K = tf.placeholder('float', shape=())
    cross_entropy = (1 - K) * adv_cross_entropy + K * normal_cross_entropy

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        np.random.seed(seed)
        step = 0
        if ada:
            K_val = lambda x: 1
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                # print(i)
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :, :, :], labels: y_train[idx, :],
                                   eps: eps_val(step), K: K_val(step), lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if ada:
                l_spec_sum = 0
                l_fit_sum = 0
                test_indices = np.random.permutation(x_test.shape[0])
                for j in range(int(x_test.shape[0] / batch_size)):
                    idx = test_indices[j * batch_size: (j + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :, :, :], labels: y_test[idx, :], eps: eps_val(step)}
                    l_spec, l_fit = sess.run([adv_cross_entropy, normal_cross_entropy], feed_dict=feed_dict_test)
                    l_spec_sum += l_spec
                    l_fit_sum += l_fit
                K_val = lambda x: l_spec_sum / (l_spec_sum + gamma * l_fit_sum)
        save(sess.run(params), name)
    tf.reset_default_graph()
    return str(time.time() - start)


if __name__ == '__main__':

    times = []

    if part == 1:  # MNIST Small
        # MNIST Parameters
        def K_val(step):
            if step <= 2000:
                return 1
            elif step <= 12000:
                return 1 - (step - 2000) / 20000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        for lr, lr_name in zip([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002],
                               ['01', '005', '002', '001', '0005', '0002']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_cnncertzero_lr_' + lr_name + '_3_' + str(e),
                                     eps_val(0.3), K_val, lr_val, cnncertzero=True, EPOCHS=e)
                times.append(t)
                print(times)

                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_cnncertzero_ada_lr_' + lr_name + '_3_' + str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e)
                times.append(t)
                print(times)
    elif part == 2:  # CIFAR Small
        # CIFAR Parameters
        def K_val(step):
            if step <= 5000:
                return 1
            elif step <= 55000:
                return 1 - (step - 5000) / 100000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 55000:
                    return eps * (step - 5000) / 50000
                else:
                    return eps

            return f


        for lr, lr_name in zip([0.0005, 0.0002], ['0005', '0002']):
            def lr_val(step):
                if step <= 60000:
                    return lr
                elif step <= 90000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [350]:
                t = train_singleprop([16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'cifar_small_singleprop_fastlin_lr_' + lr_name + '_8255_' + str(e),
                                     eps_val(8 / 255), K_val, lr_val, EPOCHS=e, batch_size=50,
                                     cifar=True, normalize=True)
                times.append(t)
                print(times)

                t = train_singleprop([16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'cifar_small_singleprop_fastlin_ada_lr_' + lr_name + '_8255_' + str(
                                         e), eps_val(8 / 255), K_val, lr_val, ada=True, EPOCHS=e,
                                     batch_size=50, cifar=True, normalize=True)
                times.append(t)
                print(times)
    elif part == 3:  # MNIST Medium
        # MNIST Parameters
        def K_val(step):
            if step <= 2000:
                return 1
            elif step <= 12000:
                return 1 - (step - 2000) / 20000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        filters = [32, 32, 64, 64, 512, 512, 10]
        kernels = [3, 4, 3, 4, 4, 1, 1]
        strides = [1, 2, 1, 2, 1, 1, 1]
        paddings = ['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID']

        for lr, lr_name in zip([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002],
                               ['01', '005', '002', '001', '0005', '0002']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_singleprop(filters, kernels, strides, paddings,
                                     'mnist_medium_singleprop_cnncertzero_lr_' + lr_name + '_3_' + str(e),
                                     eps_val(0.3), K_val, lr_val, cnncertzero=True, EPOCHS=e)
                times.append(t)
                print(times)

                t = train_singleprop(filters, kernels, strides, paddings,
                                     'mnist_medium_singleprop_cnncertzero_ada_lr_' + lr_name + '_3_' + str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e)
                times.append(t)
                print(times)
    elif part == 4:  # MNIST Wide
        # MNIST Parameters
        def K_val(step):
            if step <= 2000:
                return 1
            elif step <= 12000:
                return 1 - (step - 2000) / 20000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        filters = [128, 256, 512, 1024, 10]
        kernels = [3, 3, 3, 7, 1]
        strides = [1, 2, 2, 1, 1]
        paddings = ['SAME', 'SAME', 'SAME', 'VALID', 'SAME']

        for lr, lr_name in zip([0.001],
                               ['001']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_singleprop(filters, kernels, strides, paddings,
                                     'mnist_wide_singleprop_cnncertzero_lr_' + lr_name + '_3_' + str(e),
                                     eps_val(0.3), K_val, lr_val, cnncertzero=True, EPOCHS=e)
                times.append(t)
                print(times)
    elif part == 5:  # CIFAR Large
        # CIFAR Parameters
        def K_val(step):
            if step <= 5000:
                return 1
            elif step <= 55000:
                return 1 - (step - 5000) / 100000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 55000:
                    return eps * (step - 5000) / 50000
                else:
                    return eps

            return f


        filters = [64, 64, 128, 128, 128, 512, 10]
        kernels = [3, 3, 3, 3, 3, 16, 1]
        strides = [1, 1, 2, 1, 1, 1, 1]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'VALID', 'SAME']

        for lr, lr_name in zip([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002],
                               ['01', '005', '002', '001', '0005', '0002']):
            def lr_val(step):
                if step <= 60000:
                    return lr
                elif step <= 90000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [350]:
                t = train_singleprop(filters, kernels, strides, paddings,
                                     'cifar_large_singleprop_fastlin_lr_' + lr_name + '_8255_' + str(
                                         e), eps_val(8 / 255), K_val, lr_val, EPOCHS=e,
                                     batch_size=50, cifar=True, normalize=True)
                times.append(t)
                print(times)
    elif part == 6:  # MNIST Small - TRADES, Adv and Normal
        # MNIST Parameters
        def lr_val(step):
            if step <= 15000:
                return 0.001
            elif step <= 25000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        for e in [100]:
            t = train_normal([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                             'mnist_small_normal_' + str(e),
                             lr_val, EPOCHS=e)
            times.append(t)
            print(times)

            t = train_adv([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                          'mnist_small_adv_3_' + str(e), eps_val(0.3),
                          lr_val, EPOCHS=e)
            times.append(t)
            print(times)

            t = train_trades([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1], ['SAME', 'SAME', 'VALID', 'SAME'],
                             'mnist_small_trades_3_' + str(e), eps_val(0.3),
                             lr_val, EPOCHS=e)
            times.append(t)
            print(times)
    elif part == 7:  # MNIST Wide - Normal, Adv
        # MNIST Parameters
        def K_val(step):
            if step <= 2000:
                return 1
            elif step <= 12000:
                return 1 - (step - 2000) / 20000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        filters = [128, 256, 512, 1024, 10]
        kernels = [3, 3, 3, 7, 1]
        strides = [1, 2, 2, 1, 1]
        paddings = ['SAME', 'SAME', 'SAME', 'VALID', 'SAME']

        for lr, lr_name in zip([0.001],
                               ['001']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_normal(filters, kernels, strides, paddings,
                                 'mnist_wide_normal_lr_' + lr_name + '_' + str(e),
                                 lr_val, EPOCHS=e)
                times.append(t)
                print(times)

        for lr, lr_name in zip([0.001],
                               ['001']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_adv(filters, kernels, strides, paddings,
                              'mnist_wide_adv_lr_' + lr_name + '_3_' + str(e),
                              eps_val(0.3), lr_val, EPOCHS=e)
                times.append(t)
                print(times)
    elif part == 8:  # CIFAR Small multiple seeds
        # CIFAR Parameters
        def K_val(step):
            if step <= 5000:
                return 1
            elif step <= 55000:
                return 1 - (step - 5000) / 100000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 55000:
                    return eps * (step - 5000) / 50000
                else:
                    return eps

            return f


        filters = [32, 32, 64, 64, 512, 512, 10]
        kernels = [3, 4, 3, 4, 5, 1, 1]
        strides = [1, 2, 1, 2, 1, 1, 1]
        paddings = ['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID']

        for lr, lr_name in zip([0.0005],
                               ['0005']):
            def lr_val(step):
                if step <= 60000:
                    return lr
                elif step <= 90000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [350]:
                t = train_singleprop([16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'cifar_small_singleprop_seed_101_fastlin_ada_lr_' + lr_name + '_8255_' +
                                     str(e), eps_val(8 / 255), K_val, lr_val, ada=True, EPOCHS=e,
                                     batch_size=50, cifar=True, normalize=True, seed=101)
                times.append(t)
                print(times)

            for e in [350]:
                t = train_singleprop([16, 32, 100, 10], [4, 4, 16, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'cifar_small_singleprop_seed_102_fastlin_ada_lr_' + lr_name + '_8255_' +
                                     str(e), eps_val(8 / 255), K_val, lr_val, ada=True, EPOCHS=e,
                                     batch_size=50, cifar=True, normalize=True, seed=102)
                times.append(t)
                print(times)
    elif part == 9:  # MNIST Small Multiple seeds
        # MNIST Parameters
        def K_val(step):
            if step <= 2000:
                return 1
            elif step <= 12000:
                return 1 - (step - 2000) / 20000
            else:
                return 0.5


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 12000:
                    return eps * (step - 2000) / 10000
                else:
                    return eps

            return f


        for lr, lr_name in zip([0.0005],
                               ['0005']):
            def lr_val(step):
                if step <= 15000:
                    return lr
                elif step <= 25000:
                    return lr / 10
                else:
                    return lr / 100


            for e in [100]:
                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_seed_101_cnncertzero_ada_lr_' + lr_name + '_3_' +
                                     str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e, seed=101)
                times.append(t)
                print(times)

                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_seed_102_cnncertzero_ada_lr_' + lr_name + '_3_' +
                                     str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e, seed=102)
                times.append(t)
                print(times)

                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_seed_103_cnncertzero_ada_lr_' + lr_name + '_3_' +
                                     str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e, seed=103)
                times.append(t)
                print(times)

                t = train_singleprop([16, 32, 100, 10], [4, 4, 14, 1], [2, 1, 1, 1],
                                     ['SAME', 'SAME', 'VALID', 'SAME'],
                                     'mnist_small_singleprop_seed_104_cnncertzero_ada_lr_' + lr_name + '_3_' +
                                     str(e),
                                     eps_val(0.3), K_val, lr_val, ada=True, cnncertzero=True, EPOCHS=e, seed=104)
                times.append(t)
                print(times)
