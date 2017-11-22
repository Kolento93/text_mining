# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:30:17 2017

@desc : use vector space model
        to get a document's vector 
        then train the model
"""

import os
import re
import sys
import jieba
from sklearn.datasets import load_files
import gensim
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier 
os.chdir(sys.path[0])

def get_corpus(train_path,min_count = 50):
    """
        parameters
    --------
    train_path : str
    
    """
    print('-------- trian the corpus! --------')
    #train_path = r'source\trainset'
    train = load_files(train_path\
                       ,categories = ['C38-Politics','C39-Sports'],encoding = 'gb2312'\
                       ,decode_error = 'ignore')
    stop_words = []
    with open('stop_words.txt', 'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    stop_words = {}.fromkeys(stop_words)  
    
    pattern = re.compile(r'[\u4e00-\u9fa5]+')

    corpus = {}
    for i in range(len(train.data)):
        if i % 100 == 0:
            print(i)
        tmp = ''.join(pattern.findall(train.data[i]))
        for j in jieba.cut(tmp):
            if j not in stop_words:
                if j in corpus:
                    corpus[j] += 1
                else:
                    corpus[j] = 1
    corpus_res = {}
    for key,val in corpus.items():
        if val > min_count:
            corpus_res[key] = val
    print('length of corpus : {0}'.format(len(corpus_res)))
    
    return corpus_res

def my_vsm(data_path,corpus):
    data = load_files(data_path\
                       ,categories = ['C38-Politics','C39-Sports'],encoding = 'gb2312'\
                       ,decode_error = 'ignore')
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    #corpus = raw_corpus
    data_set = []
    for i in range(len(data.data)):
        if i % 100 == 0:
            print(i)
        signal_vec = {}.fromkeys(corpus.keys(),0)
        tmp = ''.join(pattern.findall(data.data[i]))
        for j in jieba.cut(tmp):
            if j in signal_vec:
                signal_vec[j] += 1
        data_set.append(list(signal_vec.values()))
    data_set = pd.DataFrame(data = data_set)
    data_set['label'] = data.target        
    return data_set

if __name__ == '__main__':

    train_path = 'trainset'
    test_path = 'testset'
   
    raw_corpus = get_corpus(train_path)
   
    train = my_vsm(train_path,raw_corpus)
    test = my_vsm(test_path,raw_corpus)

    xgb_model = XGBClassifier(max_depth = 50,learning_rate = 0.2,n_estimators = 20)    
    xgb_model.fit(train.ix[:,:len(raw_corpus)-1],train['label'])
    auc_train = roc_auc_score(train['label'],xgb_model.predict_proba(train.ix[:,:(len(raw_corpus) -1)])[:,1])
    auc_test = roc_auc_score(test['label'],xgb_model.predict_proba(test.ix[:,:(len(raw_corpus)-1)])[:,1])

    print('auc_train : {0}, auc_test : {1}'.format(auc_train,auc_test))





    'some changes'





