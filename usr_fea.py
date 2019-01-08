#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01
## @File: data_process2.py
## @Created Time: Mon 07 Jan 2019 11:11:13 PM CST
## @Description:
import pdb
import pandas as pd
import numpy as np
from collections import  defaultdict


df = pd.read_csv('./trian_data.csv', sep='\t', header=0)
df = df[['uid', 'fc', 'lc', 'cc', 'content']]

def func(content):
    content = content.replace("\n", "").replace(" ","")
    content = content.decode('utf-8')
    return len(content)

def func2(content):
    content = content.replace("\n","").replace(" ","")
    content = content.decode('utf-8')
    return len(set(content))

df['words_num'] = df['content'].apply(lambda x: func(x))
df['set_words_num'] = df['content'].apply(lambda x: func2(x))
df['usr_freq'] = 1
print df.head()
df.drop(['content'], axis=1, inplace=True)
res_df = df.groupby(df['uid']).sum()
res_df.reset_index(inplace=True)
res_df['usr_freq'] = res_df['usr_freq'].astype(float)
res_df['mean_words_num'] = res_df['words_num']/res_df['usr_freq']
res_df['mean_set_words_num'] = res_df['set_words_num']/res_df['usr_freq']
res_df['mean_fc'] = res_df['fc']/res_df['usr_freq']
res_df['mean_cc'] = res_df['cc']/res_df['usr_freq']
res_df['mean_lc'] = res_df['lc']/res_df['usr_freq']
#res_df.to_csv('usr_info.csv', sep='\t', header=True, index=False)
def usr_freq_func(x):
    if x > 0 and x <= 50:
        return 1
    elif x > 50 and x <= 200:
        return 2
    elif x > 200 and x <= 400:
        return 3
    elif x > 400 and x <= 700:
        return 4
    elif x > 700 and x <= 720:
        return 5
    elif x > 720 and x <= 1550:
        return 6
    elif x > 1550 and x <= 1710:
        return 7
    elif x > 1710 and x <= 2400:
        return 8
    elif x > 2400 and x <= 2750:
        return 9
    elif x > 2750 and x <= 3000:
        return 10
    elif x > 3000 and x <= 10000:
        return 11
    else:
        return 12
res_df['usr_freq'] = res_df['usr_freq'].apply(lambda x: usr_freq_func(x)) 

def mean_set_words_func(x):
    if x > 0 and x <= 10:
        return 1
    elif x > 10 and x <= 20:
        return 2
    elif x > 20 and x <= 30:
        return 3
    elif x > 30 and x <= 40:
        return 4
    elif x > 40 and x <= 50:
        return 5
    elif x > 50 and x <= 58:
        return 6
    elif x > 58 and x <= 65:
        return 7
    elif x > 65 and x <= 76:
        return 8
    elif x > 76 and x <= 85:
        return 9
    elif x > 85 and x <= 90:
        return 10
    else:
        return 11

def mean_words_func(x):
    if x > 0 and x <= 9:
        return 1
    elif x > 9 and x <= 30:
        return 2
    elif x > 30 and x <= 39:
        return 3
    elif x > 39 and x <= 50:
        return 4
    elif x > 50 and x <= 56:
        return 5
    elif x > 56 and x <= 71:
        return 6
    elif x > 71 and x <= 80:
        return 7
    elif x > 80 and x <= 95:
        return 8
    elif x > 95 and x <= 108:
        return 9
    elif x > 108 and x <= 125:
        return 10
    elif x > 125 and x <= 146:
        return 11
    elif x > 146 and x <= 160:
        return 12
    else:
        return 13

def onehot(labels, label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(label_class)] for j in range(len(labels))]) 
    return one_hot_label

res_df['mean_set_words_num'] = res_df['mean_set_words_num'].apply(lambda x: mean_set_words_func(x))
res_df['mean_words_num'] = res_df['mean_words_num'].apply(lambda x: mean_words_func(x))
res_df = res_df[['uid', 'usr_freq', 'mean_words_num', 'mean_set_words_num']]
res_df.to_csv('usr_info.csv', header=True, index=False, sep='\t')


