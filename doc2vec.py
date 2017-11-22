# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:17:49 2017

@author: haoran

@desc:
    1.使用doc2vec的编码方式将文档编码
    2.使用xgboost作为分类器进行训练

"""
import os
import re
import jieba
from sklearn.datasets import load_files
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import gensim
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier 
os.chdir(r'C:\Users\haoran\Desktop\text_mining')

#pyltp
#from sklearn.feature_extraction.text import CountVectorizer
#import jieba.posseg as pseg



def get_data(train_path,test_path):
    """
        parameters
    --------
    train_path : str
    
    test_path  : str
        
        return
    --------
    corpus_train : List(LabeledSentence)
    train.target : array
    corpus_test  : List(LabeledSentence)
    test.target  : array
    
    """

    
    train = load_files(train_path\
                       ,categories = ['C38-Politics','C39-Sports'],encoding = 'gb2312'\
                       ,decode_error = 'ignore')
    
    
    test = load_files(test_path\
                   ,categories = ['C38-Politics','C39-Sports'],encoding = 'gb2312'\
                   ,decode_error = 'ignore')
    
    #正则提取所有中文
    pattern = re.compile(r'[\u4e00-\u9fa5]+')

    
    for i in range(len(train.data)):
        train.data[i] = ''.join(pattern.findall(train.data[i]))
        
    for i in range(len(test.data)):
        test.data[i] = ''.join(pattern.findall(test.data[i]))
    
    corpus_train = []
    for i in range(len(train.data)):
        tmp_data = []
        label = 'train{0}'.format(i)
        if i % 100 == 0:
            print(label)
        for j in jieba.cut(train.data[i]):
            tmp_data.append(j)
        corpus_train.append(LabeledSentence(tmp_data,[label]))
    
    corpus_test = []
    for i in range(len(test.data)):
        tmp_data = []
        label = 'test{0}'.format(i)
        if i % 100 == 0:
            print(label)
        for j in jieba.cut(test.data[i]):
            tmp_data.append(j)
        corpus_test.append(LabeledSentence(tmp_data,[label]))
        
    return corpus_train,train.target,corpus_test,test.target
    
    
def data2vec(data,target,model):
    data_vec = []
    for i in range(len(data)):
        data_vec.append(model.docvecs[data[i].tags[0]])
        
    data_vec = pd.DataFrame(data_vec)  
    data_vec['label'] = target
    
    return data_vec

def train__model_get_result(train,train_y,test,test_y):
    corpus = train.copy()
    corpus.extend(test)
    model = Doc2Vec(corpus,size = 100,window = 5,\
                min_count = 5,workers=4)
    
    train_vec = data2vec(train,train_y,model)
    test_vec = data2vec(test,test_y,model)
    
    xgb_model = XGBClassifier(max_depth = 5,learning_rate = 0.05,n_estimators = 300)    
    xgb_model.fit(train_vec.ix[:,0:100],train_vec['label'])
    auc_train = roc_auc_score(train_vec['label'],xgb_model.predict_proba(train_vec.ix[:,0:100])[:,1])
    auc_test = roc_auc_score(test_vec['label'],xgb_model.predict_proba(test_vec.ix[:,0:100])[:,1])
    
    print('train_auc : {0} , test_auc : {1}'.format(auc_train,auc_test))
    return    
    


if __name__ == '__main__':
    train_path = r'source\trainset'
    test_path = r'source\testset'
    train,train_y,test,test_y = get_data(train_path,test_path)
    train__model_get_result(train,train_y,test,test_y)
    










