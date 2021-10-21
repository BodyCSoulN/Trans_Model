# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     TransH_model
    @Description:   a tensorflow implemention of TransH
    @Author:        lzx
    @Time:          2021/10/12 15:42
------------------------------------------------------------
"""
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
        self._build_input(args.dim, n_head, n_relation, n_tail)
        self._build_model()

    def _build_input(self, dim, n_head, n_relation, n_tail):
        self.head_indices = tf.placeholder(tf.float32, [None], 'head_indices')
        self.relation_indices = tf.placeholder(tf.float32, [None], 'relation_indices')
        self.tail_indices = tf.placeholder(tf.float32, [None], 'tail_indices')
        self.dim = dim
        self.n_head = n_head
        self.n_relation = n_relation
        self.n_tail = n_tail

    def _build_model(self):
        self.head_emb_matrix = tf.get_variable('head_emb_matrix', [self.n_head, self.dim])
        self.relation_emb_matrix = tf.get_variable('relation_emb_matrix', [self.n_relation, self.dim])
        self.tail_emb_matrix = tf.get_variable('tail_emb_matrix', [self.n_tail, self.dim])

        self.head_embedding = tf.nn.embedding_lookup(self.head_emb_matrix, self.head_indices)
        self.relation_embedding = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        self.tail_embedding = tf.nn.embedding_lookup(self.tail_emb_matrix, self.tail_indices)

    def _build_train(self, n_epochs):
        print("start to train...")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        for epoch in n_epochs:
            pass
        # TODO

    def _build_loss(self, margin):

