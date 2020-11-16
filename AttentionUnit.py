#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:35
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class AttentionWrapper(object):
    def __init__(self, hidden_size, input_size, hs, scope_name):
        self.hs = tf.compat.v1.transpose(hs, [1,0,2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}

        with tf.compat.v1.variable_scope(scope_name):
            self.Wh = tf.compat.v1.get_variable('Wh', [input_size, hidden_size])
            self.bh = tf.compat.v1.get_variable('bh', [hidden_size])
            self.Ws = tf.compat.v1.get_variable('Ws', [input_size, hidden_size])
            self.bs = tf.compat.v1.get_variable('bs', [hidden_size])
            self.Wo = tf.compat.v1.get_variable('Wo', [2*input_size, hidden_size])
            self.bo = tf.compat.v1.get_variable('bo', [hidden_size])
        self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
                            'bh': self.bh, 'bs': self.bs, 'bo': self.bo})

        hs2d = tf.compat.v1.reshape(self.hs, [-1, input_size])
        phi_hs2d = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        self.phi_hs = tf.compat.v1.reshape(phi_hs2d, tf.compat.v1.shape(self.hs))

    def __call__(self, x, finished = None):
        gamma_h = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(x, self.Ws, self.bs))
        weights = tf.compat.v1.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
        weight = weights
        weights = tf.compat.v1.exp(weights - tf.compat.v1.reduce_max(weights, reduction_indices=0, keep_dims=True))
        weights = tf.compat.v1.divide(weights, (1e-6 + tf.compat.v1.reduce_sum(weights, reduction_indices=0, keep_dims=True)))
        context = tf.compat.v1.reduce_sum(self.hs * weights, reduction_indices=0)
        # print wrt.get_shape().as_list()
        out = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(tf.compat.v1.concat([context, x], -1), self.Wo, self.bo))

        if finished is not None:
            out = tf.compat.v1.where(finished, tf.compat.v1.zeros_like(out), out)
        return out, weights

    def save(self, path):
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        param_values = pickle.load(open(path, 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])