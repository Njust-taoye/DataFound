#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01
## @File: feature.py
## @Created Time: Thu 27 Dec 2018 01:33:40 PM CST
## @Description:
import pdb
import copy,os,sys,psutil
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool,Manager
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import datetime
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer
import zhon
from zhon import hanzi
from zhon.hanzi import punctuation


def remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
    document = re.sub(noise_pattern, "", document)
    symbol_patt = re.compile(r'[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    document = re.sub(symbol_patt, "", document)
    document = re.sub(ur"[%s]+" %punctuation, " ", document)
    patt = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    document = re.sub(patt, "", document)
    eng_punctuation = '!,;\.:\(\)\{\}\[\]\+\=\-_\<\>\*&\^%$#@!~?"\''
    document = re.sub(r'[{}]+'.format(eng_punctuation),' ',document)
    return document

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


def cal_score(real_fc, pred_fc):
    real_fc = np.reshape(real_fc, (-1, ))
    dev_f = np.abs(pred_fc - real_fc)/(real_fc + 5) 
    dev = 1 - dev_f
    def func(x):
        if x - 0.8 > 0:
            return 1
        else:
            return 0
    dev = np.array([func(x) for x in dev])
    count = real_fc + 1
    def func2(x):
        if x > 50:
            return 50
        else:
            return x
    count = np.array([func2(x) for x in count])
    count = np.reshape(count, (-1,))
    dev = np.reshape(dev, (-1,))
    res = sum(count * dev)/ float(sum(count))
    return res

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def tune_model(train_X, train_Y):
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    #tune_params = {'n_estimators': range(100, 500,100), 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': range(50, 500, 50), \
    #        'min_samples_split': [2, 4, 6, 10, 20], 'min_samples_leaf': [1, 2, 3, 5, 10]} 
    
    tune_params = {'n_estimators': [100, 500], 'max_features': ['sqrt', 'log2'], 'max_depth': [100, 200]}
    score = make_scorer(cal_score, greater_is_better=False)
    gsearch = GridSearchCV(estimator = RandomForestRegressor(oob_score=False, random_state=10), param_grid = tune_params,\
                                                            scoring=score, cv=5)
    gsearch.fit(train_X, np.array(train_Y))
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    print ("Best score: %0.3f" % gsearch.best_score_)
    print ("Best paramters set:")
    best_paramters = gsearch.best_estimator_.get_params()
    for param_name in sorted(tune_params.keys()):
        print("\t%s: %r" % (param_name, best_paramters[param_name]))

def feature_discretization(fea):
    if fea > 0 and fea <= 4:
        return 0
    elif fea > 4 and fea <= 7:
        return 1
    elif fea > 7 and fea <= 10:
        return 2
    elif fea > 10 and fea <= 14:
        return 3
    elif fea > 14 and fea <= 19:
        return 4
    elif fea > 19 and fea <= 25:
        return 5
    elif fea > 25 and fea <= 31:
        return 6
    elif fea > 31 and fea <= 33:
        return 7
    elif fea > 33 and fea <= 37:
        return 8
    elif fea > 37 and fea <= 40:
        return 9
    elif fea > 40 and fea <= 43:
        return 10
    elif fea > 43 and fea <= 50:
        return 11
    elif fea > 50 and fea <= 65:
        return 12
    elif fea > 65 and fea <= 70:
        return 13
    elif fea > 70 and fea <= 80:
        return 14
    elif fea > 80 and fea <= 91:
        return 15
    elif fea > 91 and fea <= 98:
        return 16
    elif fea > 98 and fea <= 103:
        return 17
    else:
        return 18


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--addWeekday", help="add weekday feature if True", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--NGram_num", help="NGram feature num", type=int)
    parser.add_argument("--tune_mode", help="tune_model if True", type=str2bool, nargs="?", const=True, default=False)
    args = parser.parse_args()
    print "是否增加weekday特征", args.addWeekday
    print "文本特征维度", args.NGram_num
    print "是否在调参模式", args.tune_mode
    filename = './trian_data.csv'
    df = pd.read_csv(filename, sep='\t', header=0)
    df['fc'] = df['fc'].astype(float)
    df['cc'] = df['cc'].astype(float)
    df['lc'] = df['lc'].astype(float)
    df.dropna(inplace=True)
    #train_df = df[(df['time'] > "2015-03-10 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    #eval_df = df[(df['time'] >= "2015-07-01 00:00:00")]
    train_df = df[(df['time'] > "2015-06-25 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
    eval_df = df[(df['time'] >= "2015-07-01 00:00:00") & (df['time'] <= '2015-07-03 00:00:00')]
    print "train and eval has splited!!!!"
    weekday_fea = args.addWeekday
    if weekday_fea:
        train_df['weekday'] = train_df['time'].apply(lambda x: get_weekday(x))
        train_dayhot = onehot(list(train_df['weekday']), 7)
        train_dayhot = pd.DataFrame(train_dayhot, columns=range(7))
        train_df.reset_index(drop=True, inplace=True)#两表合并时，两表最好都做如此操作
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
    train_words_num = []
    train_set_words_num = []
    def func(i):
        line = train_df.loc[i]
        content = line['content'].replace("\n", " ")
        words_num = len(content.replace(" ","").decode('utf-8'))
        set_words_num = len(set(content.replace(" ","").decode('utf-8')))
        word_cut = jieba.cut(content, cut_all=False)
        word_list = list(word_cut)
        word_list = ' '.join(word_list)
        res = []
        res.append(word_list)
        res.append(str(words_num))
        res.append(str(set_words_num))
        return '\t'.join(res)
    if not os.path.exists('cuted_word.txt'):
        print "cuted_word.txt not exists !!!"
        rst = []
        pool = Pool(8)
        for i in range(len(train_df)):
            rst.append(pool.apply_async(func, args=(i,)))
        pool.close()
        pool.join()
        print "cuted !!!!!!!!"
        rst = [i.get() for i in rst]
        with open('cuted_word.txt', 'w') as fp:
            for i in rst:
                fp.write(i+'\n')
                content, words_num, set_words_num = i.strip().split('\t')
                data_list.append(content)
                train_words_num.append(int(words_num))
                train_set_words_num.append(int(set_words_num))
        fp.close()
    else:
        print "cuted_word.txt has exists!!!!!"
        with open('cuted_word.txt', 'r') as rp:
            line = rp.readline()
            while line:
                content, words_num, set_words_num = line.strip().split('\t')
                train_words_num.append(int(words_num))
                train_set_words_num.append(int(set_words_num))
                data_list.append(content)
                line = rp.readline()
        rp.close()

    
    NGram_fea_num = args.NGram_num
    vectorizer = CountVectorizer(min_df=1, max_features=NGram_fea_num, ngram_range=(1,2), analyzer = 'word', 
                                stop_words = get_stop_words(), token_pattern='\D+')
    train_nGram = vectorizer.fit_transform(data_list).toarray()
    if not os.path.exists(str(NGram_fea_num)+"words.txt"):
        with open(str(NGram_fea_num)+"words.txt", 'w') as fp:
            for word in vectorizer.get_feature_names():
                fp.write(word+"\n")
    #pdb.set_trace()
    train_words_num = pd.DataFrame(train_words_num, columns=['words_num'])
    train_set_words_num = pd.DataFrame(train_set_words_num, columns=['set_words_num'])
    train_set_words_num['set_words_num'] = train_set_words_num['set_words_num'].apply(lambda x: feature_discretization(x))
    train_set_words_onehot = onehot(list(train_set_words_num['set_words_num']), 19)
    train_set_words_num = pd.DataFrame(train_set_words_onehot, columns=range(19))
    train_nGram = pd.DataFrame(train_nGram, columns=range(NGram_fea_num))
    train_words_num.reset_index(drop=True, inplace=True)
    train_nGram.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    train_set_words_num.reset_index(drop=True, inplace=True)
    train_df = pd.concat([train_df, train_words_num, train_set_words_num, train_nGram], axis=1)
    #train_df['set_words_num'] = train_df['set_words_num'].apply(lambda x: feature_discretization(x))
    print "Get Train NGram feature!!!!"
    train_Y = train_df['fc'].ravel()
    train_X = train_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
    tune_mode = args.tune_mode
    if not tune_mode:
        print "model established!!!!!"
        rf = RandomForestRegressor(oob_score=False, random_state=10)
        #rf = GradientBoostingRegressor(random_state=10)
        #rf = RandomForestRegressor(n_estimators=500, max_features='log', max_depth=100, \
        #                            min_samples_split=4, min_samples_leaf=2)
        print rf.fit(train_X, train_Y)
    if tune_mode:
        print "tune model start !!!!!"
        tune_model(train_X, train_Y)
    print "model fited!!!!!"
    if not tune_mode: 
        eval_words_num = [] 
        eval_data_list = [] 
        eval_set_words_num = [] 
        def func(i):
            line = eval_df.loc[i]
            content = line['content'].replace("\n", " ")
            words_num = len(content.replace(" ", "").decode('utf-8'))
            set_words_num = len(set(content.replace(" ","").decode('utf-8')))
            word_cut = jieba.cut(content, cut_all=False)
            word_list = list(word_cut)
            word_list = ' '.join(word_list)
            res = [] 
            res.append(word_list)
            res.append(str(words_num))
            res.append(str(set_words_num))
            return '\t'.join(res)
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
                    content, words_num, set_words_num = i.strip().split('\t')
                    eval_data_list.append(content)
                    eval_words_num.append(int(words_num))
                    eval_set_words_num.append(int(set_words_num))
                    fp.write(i+'\n')
                    #fp.write('\n')
            fp.close()
        else:
            print "eval_words has exists !!!!!" 
            with open('eval_words.txt', 'r') as erp:
                line = erp.readline()
                while line:
                    content, words_num, set_words_num = line.strip().split('\t')
                    eval_data_list.append(content)
                    eval_words_num.append(int(words_num))
                    eval_set_words_num.append(int(set_words_num))
                    line = erp.readline()
            erp.close()
        eval_nGram = vectorizer.transform(eval_data_list).toarray()
        eval_nGram = pd.DataFrame(eval_nGram, columns=range(NGram_fea_num))
        eval_words_num = pd.DataFrame(eval_words_num, columns=['words_num'])
        eval_set_words_num = pd.DataFrame(eval_set_words_num, columns=['set_words_num'])
        eval_set_words_num['set_words_num'] = eval_set_words_num['set_words_num'].apply(lambda x: feature_discretization(x))
        eval_set_words_onehot = onehot(list(eval_set_words_num['set_words_num']), 19)
        eval_set_words_num = pd.DataFrame(eval_set_words_onehot, columns=range(19))
        eval_words_num.reset_index(inplace=True, drop=True)
        eval_nGram.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)
        eval_set_words_num.reset_index(drop=True, inplace=True)
        print "Get eval NGram feature!!!!!"
        eval_df = pd.concat([eval_df, eval_words_num, eval_set_words_num, eval_nGram], axis=1) 
        #eval_df['set_words_num'] = eval_df['set_words_num'].apply(lambda x: feature_discretization(x))
        eval_Y = np.array(eval_df[['fc']])
        eval_X = eval_df.drop(['uid', 'mid', 'time', 'content', 'fc', 'cc', 'lc'], axis=1)
        pred_Y = rf.predict(eval_X)
        print cal_score(eval_Y, pred_Y)

