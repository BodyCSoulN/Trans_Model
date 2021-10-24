# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     TransH_model
    @Description:   a tensorflow implemention of Trans model.
    @Author:        lzx
    @Time:          2021/10/12 15:42
------------------------------------------------------------
"""
import math

import tensorflow as tf


class TransH(object):
    def __init__(self):
        self._build_input()
        self._build_model()
        self._build_loss()
        self._build_train()

    # def _build_input(self):


class TransE(object):
    def __init__(self, args, n_head, n_relation, n_tail):
        self._build_input(n_head, n_relation, n_tail)
        self._build_model(args)
        self._build_loss(args)
        self._build_train(args)

    def _build_input(self, n_head, n_relation, n_tail):
        # 注意 indices 为int  不能为float
        self.pos_head_indices = tf.placeholder(tf.int32, [None], 'pos_head_indices')
        self.pos_relation_indices = tf.placeholder(tf.int32, [None], 'pos_relation_indices')
        self.pos_tail_indices = tf.placeholder(tf.int32, [None], 'pos_tail_indices')

        self.neg_head_indices = tf.placeholder(tf.int32, [None], 'neg_head_indices')
        self.neg_relation_indices = tf.placeholder(tf.int32, [None], 'neg_relation_indices')
        self.neg_tail_indices = tf.placeholder(tf.int32, [None], 'neg_tail_indices')
        self.n_head = n_head
        self.n_relation = n_relation
        self.n_tail = n_tail

    def _build_model(self, args):
        bound = 6 / math.sqrt(args.dim)
        self.head_emb_matrix = tf.get_variable('head_emb_matrix', [self.n_head, args.dim],
                                               initializer=tf.random_uniform_initializer(-bound, bound))
        self.relation_emb_matrix = tf.get_variable('relation_emb_matrix', [self.n_relation, args.dim],
                                                   initializer=tf.random_uniform_initializer(-bound, bound))
        self.tail_emb_matrix = tf.get_variable('tail_emb_matrix', [self.n_tail, args.dim],
                                               initializer=tf.random_uniform_initializer(-bound, bound))
        # [batch_size, dim]
        self.pos_h = tf.nn.embedding_lookup(self.head_emb_matrix, self.pos_head_indices)
        self.pos_r = tf.nn.embedding_lookup(self.relation_emb_matrix, self.pos_relation_indices)
        self.pos_t = tf.nn.embedding_lookup(self.tail_emb_matrix, self.pos_tail_indices)

        self.neg_h = tf.nn.embedding_lookup(self.head_emb_matrix, self.neg_head_indices)
        self.neg_r = tf.nn.embedding_lookup(self.relation_emb_matrix, self.neg_relation_indices)
        self.neg_t = tf.nn.embedding_lookup(self.tail_emb_matrix, self.neg_tail_indices)

    def _build_train(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train_kge(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def _build_loss(self, args):
        pos_d, neg_d = self._caculate_d(args)
        every_loss = tf.nn.relu(pos_d - neg_d + args.margin)
        # (batch_size,) -> a number
        self.loss = tf.reduce_sum(every_loss)

    def _caculate_d(self, args):
        if args.L1_flag:
            # [batch_size, dim] -> 一维数组
            pos_d = tf.reduce_sum(tf.abs(self.pos_h + self.pos_r - self.pos_t), axis=1)
            neg_d = tf.reduce_sum(tf.abs(self.neg_h + self.neg_h - self.neg_t), axis=1)

        else:
            # 计算L2距离 pos_h [batch_size, dim] ->一维数组
            pos_d = tf.sqrt(tf.reduce_sum(tf.square(self.pos_h + self.pos_r - self.pos_t), axis=1))
            neg_d = tf.sqrt(tf.reduce_sum(tf.square(self.neg_h + self.neg_r - self.neg_t), axis=1))
        return pos_d, neg_d



