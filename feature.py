#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01
## @Mail: taoye01@baidu.com
## @File: get_nGram.py
## @Copyright (c) 2012-2018 Baidu.com, Inc. All Rights Reserved
## @Created Time: Thu 27 Dec 2018 01:33:40 PM CST
## @Description:
import pdb
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import nltk
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from multiprocessing import Pool,Manager
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime
import argparse


def _remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
    clean_text = re.sub(noise_pattern, "", document)
    return clean_text

def get_stop_words():
    stop_words = []
    with open('stopwords_cn.txt') as fp:
        for line in fp.readlines():
            stop_word = line.strip().decode('utf-8')
            stop_words.append(stop_word)
    return stop_words

def get_weekday(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday()

def onehot(labels, label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(label_class)] for j in range(len(labels))]) 
    return one_hot_label

def cal_score(real_Y, pred_Y):
    real_fc = real_Y['fc']
    real_cc = real_Y['cc']
    real_lc = real_Y['lc']

    pred_fc = pred_Y['fc']
    pred_cc = pred_Y['cc']
    pred_lc = pred_Y['lc']
    dev_f = np.abs(pred_fc - real_fc)/(pred_fc + 5) 
    dev_c = np.abs(pred_cc - real_cc)/(pred_cc + 3)
    dev_l = np.abs(pred_lc - real_lc)/(pred_lc + 3)
    dev = 1 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l
    dev = dev.apply(lambda x: 1 if (x-0.8) > 0 else 0)

    count = real_fc + real_cc + real_lc + 1
    count = count.apply(lambda x: 100 if x > 100 else x)
    res = sum(count * dev)/ float(sum(count))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--addWeekday", help="add weekday feature if True", default=True)
    parser.add_argument("--NGram_num", help="NGram feature num", type=int)
    args = parser.parse_args()

    filename = 'weibo_train_data.txt'
    df = pd.read_csv(filename, sep='\t', header=None, names=['uid', 'mid', 'time', 'fc', 'cc', 'lc', 'content'])
    df.dropna(inplace=True)
    #train_df = df[(df['time'] > "2015-03-10 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    #eval_df = df[(df['time'] >= "2015-07-01 00:00:00")]
    train_df = df[(df['time'] > "2015-06-25 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    eval_df = df[(df['time'] >= "2015-07-29 00:00:00")]

    print "train and eval has splited!!!!"

    weekday_fea = args.addWeekday
    if weekday_fea:
        train_df['weekday'] = train_df['time'].apply(lambda x: get_weekday(x))
        train_dayhot = onehot(list(train_df['weekday']), 7)
        train_dayhot = pd.DataFrame(train_dayhot, columns=range(7))
        train_df.reset_index(drop=True, inplace=True)
        train_dayhot.reset_index(drop=True, inplace=True)
        train_df = pd.concat([train_df, train_dayhot], axis=1)
        train_df.drop(['weekday'], axis=1, inplace=True)

        eval_df['weekday'] = eval_df['time'].apply(lambda x: get_weekday(x))
        eval_dayhot = onehot(list(eval_df['weekday']), 7)
        eval_dayhot = pd.DataFrame(eval_dayhot, columns=range(7))
        eval_dayhot.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)
        eval_df = pd.concat([eval_df, eval_dayhot], axis=1)
        eval_df.drop(['weekday'], axis=1, inplace=True)
        print "weekday onehoted!!!" 

    data_list = [] 
    def func(i):
        line = train_df.loc[i]
        content = line['content'].replace("\n", " ")
        word_cut = jieba.cut(content, cut_all=False)
        word_list = list(word_cut)
        return ' '.join(word_list)
    if not os.path.exists('cuted_word.txt'):
        print "cuted_word.txt not exists !!!"
        rst = []
        pool = Pool(8)
        for i in tqdm(range(len(train_df))):
            rst.append(pool.apply_async(func, args=(i,)))
        pool.close()
        pool.join()
        print "cuted !!!!!!!!"
        rst = [i.get() for i in rst]
        with open('cuted_word.txt', 'w') as fp:
            for i in rst:
                fp.write(i+'\n')
                data_list.append(i)
        fp.close()
    else:
        print "cuted_word.txt has exists!!!!!"
        with open('cuted_word.txt', 'r') as rp:
            line = rp.readline()
            while line:
                data_list.append(line)
                line = rp.readline()
        rp.close()

    
    NGram_fea_num = args.NGram_num
    vectorizer = CountVectorizer(min_df=1, max_features=NGram_fea_num, ngram_range=(1,2), analyzer = 'word', stop_words = get_stop_words(), preprocessor=_remove_noise)
    train_nGram = vectorizer.fit_transform(data_list).toarray()
    train_nGram = pd.DataFrame(train_nGram, columns=range(NGram_fea_num))
    train_nGram.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    train_df = pd.concat([train_df, train_nGram], axis=1)
    print "Get Train NGram feature!!!!"
    train_Y = train_df[['fc', 'cc', 'lc']]
    train_X = train_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
    print "model established!!!!!"
    rf = RandomForestRegressor(oob_score=False, random_state=10)
    rf.fit(train_X, train_Y)
    print "model fited!!!!!"
     
    eval_data_list = [] 
    def func(i):
        line = eval_df.loc[i]
        content = line['content'].replace("\n", " ")
        word_cut = jieba.cut(content, cut_all=False)
        word_list = list(word_cut)
        return ' '.join(word_list)
    if not os.path.exists('eval_words.txt'):
        print "eval_words.txt not exists!!!!!"
        rst = []
        pool = Pool(8)
        for i in range(len(eval_df)):
            rst.append(pool.apply_async(func, args=(i,)))
        pool.close()
        pool.join()
        rst = [i.get() for i in rst]
        with open('eval_words.txt', 'w') as fp:
            for i in rst:
                eval_data_list.append(i)
                fp.write(i+'\n')
                #fp.write('\n')
        fp.close()
    else:
        print "eval_words has exists !!!!!" 
        with open('eval_words.txt', 'r') as erp:
            line = erp.readline()
            while line:
                eval_data_list.append(line)
                line = erp.readline()
        erp.close()

    eval_nGram = vectorizer.transform(eval_data_list).toarray()
    eval_nGram = pd.DataFrame(eval_nGram, columns=range(NGram_fea_num))
    eval_nGram.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    print "Get eval NGram feature!!!!!"
    eval_df = pd.concat([eval_df, eval_nGram], axis=1) 
    eval_Y = eval_df[['fc', 'cc', 'lc']]
    eval_X = eval_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
    pred_Y = rf.predict(eval_X)
    pred_Y = pd.DataFrame(pred_Y, columns=['fc', 'cc', 'lc'])
    print cal_score(eval_Y, pred_Y)

