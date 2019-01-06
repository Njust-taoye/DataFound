#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import os
import numpy as np
import re
import zhon
from zhon import hanzi
#from zhon.hanzi import punctuation
import string

df = pd.read_csv('./weibo_train_data.txt', sep='\t',  header=None, names=['uid','mid', 'time', 'fc', 'cc', 'lc', 'content'])
df.dropna(inplace=True)
print df.head()
eval_df = pd.read_csv('./weibo_predict_data.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'content'])
eval_df.dropna(inplace=True)
print eval_df.head()

def remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+", "\<.*\>"]))
    document = re.sub(noise_pattern, "", document)
    symbol_patt = re.compile(r'[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    document = re.sub(symbol_patt, "", document)
    punctuation = '！？｡＂\。＃＄％＆＇ヽ∀ ﾉ🍃《（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》★「」/『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
    document = re.sub(ur"[%s]+" %punctuation, " ", document)
    patt = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    document = re.sub(patt, "", document)
    ##eng_punctuation = '!,;\.:\(\)\{\}\[\]\+\=\-_\<\>\*&\^%$#@!~?"\'<200b>'
    eng_punctuation = string.punctuation
    document = re.sub(r'[{}]+'.format(eng_punctuation),' ',document)
    if document.strip() == '':
        return np.nan
    return document

df['content'] = df['content'].apply(lambda x: remove_noise(x.decode('utf-8')))
df.dropna(inplace=True)

eval_df['content'] = eval_df['content'].apply(lambda x: remove_noise(x.decode('utf-8')))
eval_df.dropna(inplace=True)

eval_df.to_csv('pred_data.csv', sep='\t', columns=['uid', 'mid', 'time', 'content'], header=True, index=None)
df.to_csv('trian_data.csv', sep='\t', columns=['uid', 'mid', 'time', 'fc', 'cc', 'lc', 'content'], header=True, index=None)
