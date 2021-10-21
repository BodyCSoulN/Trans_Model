# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     main.py
    @Description:   
    @Author:        lzx
    @Time:          2021/10/12 17:06
------------------------------------------------------------
"""
import Trans_model
import preprocess

if __name__ == '__main__':
    datapath = "./data/FB15k"
    data = preprocess.convert_kg2id(datapath)
