# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     train.py
    @Description:   training model
    @Author:        lzx
    @Time:          2021/10/21 16:45
------------------------------------------------------------
"""
import random

import tensorflow as tf
import numpy as np
import Trans_model


def train(data, args, show_loss):
    kg_np, n_head, n_relation, n_tail = data[0], data[1], data[2], data[3]
    corrupted_np = corrupted_triple(kg_np)
    model = Trans_model.TransE(args, n_head, n_relation, n_tail)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.epochs):
            # np.random.shuffle(kg_np)
            print("start to train transE")
            start = 0
            while start < kg_np.shape[0]:
                _, loss = model.train_kge(sess, get_feed_dict(model, kg_np, corrupted_np, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print("the training loss = {}".format(loss))
                    # print("the shape of loss = {}".format(loss.shape))
    # todo test


def get_feed_dict(model, kg_np, corrupted_np, start, end):
    kge_feed_dict = {
        model.pos_head_indices: kg_np[start:end, 0],
        model.pos_tail_indices: kg_np[start:end, 1],
        model.pos_relation_indices: kg_np[start:end, 2],
        model.neg_head_indices: corrupted_np[start:end, 0],
        model.neg_tail_indices: corrupted_np[start:end, 1],
        model.neg_relation_indices: corrupted_np[start:end, 2]
    }
    return kge_feed_dict


def corrupted_triple(kg_np):
    """
        将数据处理为正例和负例：
        正例：在知识图谱中存在的三元组
        负例：将三元组替换头实体/尾实体后得到的三元组
    """
    corrupted_list = kg_np[:]
    entity = set(kg_np[:, 0]) | set(kg_np[:, 1])
    for triple in corrupted_list:
        i = random.uniform(-1, 1)
        if i < 0:
            rand_entity = triple[0]
            while rand_entity == triple[0]:
                # 这里注意sample返回的是一个列表
                rand_entity = random.sample(entity, 1)[0]
            # 随机替换头实体 从除了该实体以外的其他实体中随机选择一个实体(包括头 尾)
            triple[0] = rand_entity
        else:
            # 随机替换尾实体
            rand_entity = triple[1]
            while rand_entity == triple[1]:
                rand_entity = random.sample(entity, 1)[0]
            triple[1] = rand_entity
    return corrupted_list


def eval(self, kg_np, ):
    # 对于每一个测试集中的三元组，将头实体依次替换，使用模型计算其相似度d(h + r, t)，升序排列。
    # (d越小，代表模型预测的该三元组越可能正确)
    # 存储正确三元组的排序
    correct_rank = dict()
    triple_replace_d = dict()
    for triple in kg_np:
        pass
    # todo





