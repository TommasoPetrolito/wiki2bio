#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-12 下午10:47
# @Author  : Tianyu Liu

import tensorflow as tf
import pickle


class dualAttentionWrapper(object):
    def __init__(self, hidden_size, input_size, field_size, hs, fds, scope_name):
        self.hs = tf.compat.v1.transpose(hs, [1,0,2])  # input_len * batch * input_size
        self.fds = tf.compat.v1.transpose(fds, [1,0,2])
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
            self.Wf = tf.compat.v1.get_variable('Wf', [field_size, hidden_size])
            self.bf = tf.compat.v1.get_variable('bf', [hidden_size])
            self.Wr = tf.compat.v1.get_variable('Wr', [input_size, hidden_size])
            self.br = tf.compat.v1.get_variable('br', [hidden_size])

        self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
                            'bh': self.bh, 'bs': self.bs, 'bo': self.bo,
                            'Wf': self.Wf, 'Wr': self.Wr, 
                            'bf': self.bf, 'br': self.br})

        hs2d = tf.compat.v1.reshape(self.hs, [-1, input_size])
        phi_hs2d = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        self.phi_hs = tf.compat.v1.reshape(phi_hs2d, tf.compat.v1.shape(self.hs))
        fds2d = tf.compat.v1.reshape(self.fds, [-1, field_size])
        phi_fds2d = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(fds2d, self.Wf, self.bf))
        self.phi_fds = tf.compat.v1.reshape(phi_fds2d, tf.compat.v1.shape(self.hs))

    def __call__(self, x, coverage = None, finished = None):
        gamma_h = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(x, self.Ws, self.bs))  # batch * hidden_size
        alpha_h = tf.compat.v1.tanh(tf.compat.v1.nn.xw_plus_b(x, self.Wr, self.br))
        fd_weights = tf.compat.v1.reduce_sum(self.phi_fds * alpha_h, reduction_indices=2, keep_dims=True)
        fd_weights = tf.compat.v1.exp(fd_weights - tf.compat.v1.reduce_max(fd_weights, reduction_indices=0, keep_dims=True))
        fd_weights = tf.compat.v1.divide(fd_weights, (1e-6 + tf.compat.v1.reduce_sum(fd_weights, reduction_indices=0, keep_dims=True)))
        
        
        weights = tf.compat.v1.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)  # input_len * batch
        weights = tf.compat.v1.exp(weights - tf.compat.v1.reduce_max(weights, reduction_indices=0, keep_dims=True))
        weights = tf.compat.v1.divide(weights, (1e-6 + tf.compat.v1.reduce_sum(weights, reduction_indices=0, keep_dims=True)))
        weights = tf.compat.v1.divide(weights * fd_weights, (1e-6 + tf.compat.v1.reduce_sum(weights * fd_weights, reduction_indices=0, keep_dims=True)))
        
        context = tf.compat.v1.reduce_sum(self.hs * weights, reduction_indices=0)  # batch * input_size
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
