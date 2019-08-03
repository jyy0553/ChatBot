# import pickle as pk
# inf,a = pk.load(open("dataset-cornell-length10-filter1-vocabSize40000.pkl"))
import os
import nltk
import numpy as np
import pickle
import random

def loadDataset(filename):
    """
    :param filename: 数据的路径，数据是一个json结构，包含三部分，分别是word2id，即word到id的转换，
    id2word，即id到word的转换 ，以及训练数据trainingSamples，是一个二维数组，形状为N*2，每一行包含问题和回答
    :return: 通过pickle解析我们的数据，返回上述的三部分内容。
    """
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    print(word2id)
    print(id2word)
    print(trainingSamples)
    return word2id, id2word, trainingSamples

a,b,c = loadDataset("dataset-cornell-length10-filter1-vocabSize40000.pkl")