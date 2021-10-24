# -*- coding:utf-8 -*-
"""
------------------------------------------------------------
    @File Name:     main.py
    @Description:   
    @Author:        lzx
    @Time:          2021/10/12 17:06
------------------------------------------------------------
"""
import preprocess
import argparse
from train import train

if __name__ == '__main__':
    datapath = "./data/FB15k/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=int, default=0.01, help="learning rate of trans model")
    parser.add_argument("--dim", type=int, default=50, help="dimension of entity embeddings")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--epochs", type=int, default=20, help="the training epochs")
    parser.add_argument("--margin", type=int, default=1, help="the ranking margin")
    parser.add_argument("--L1_flag", type=bool, default=True, help="whether use L1")
    args = parser.parse_args()
    data = preprocess.convert_kg2id(datapath)
    train(data, args, True)

