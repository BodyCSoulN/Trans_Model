# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     data_loader
    @Description:   preprocess of data
    @Author:        lzx
    @Time:          2021/10/12 17:09
------------------------------------------------------------
"""
import numpy as np

def convert_kg2id(datapath):
    entity2id = dict()
    relation2id = dict()

    entity2id_file = open(datapath + "entity2id.txt", "r", encoding="utf-8")
    relation2id_file = open(datapath + "relation2id.txt", "r", encoding="utf-8")
    testdata2id_file = open(datapath + "test.txt", "r", encoding="utf-8")
    for line in entity2id_file:
        array = line.strip().split("\t")
        entity2id[array[0]] = array[1]

    for line in relation2id_file:
        array = line.strip().split("\t")
        relation2id[array[0]] = array[1]

    tmp_list = []
    for line in testdata2id_file:
        array = line.strip().split("\t")
        if array[0] not in entity2id or array[1] not in entity2id or array[2] not in relation2id:
            continue
        headid = entity2id[array[0]]
        tailid = entity2id[array[1]]
        relationid = relation2id[array[2]]
        tmp_list.append([headid, tailid, relationid])

    kg_np = np.array(tmp_list)
    n_head = len(set(kg_np[:, 0]))
    n_tail = len(set(kg_np[:, 1]))
    n_relation = len(set(kg_np[:, 2]))
    return kg_np, n_head, n_tail, n_relation
