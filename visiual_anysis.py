#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: taoye01


import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import argparse


parse = argparse.ArgumentParser()
parse.add_argument("-f","--Filename", help="need to plot filename", type=str)
parse.add_argument("-x", "--X_var", help="X coordinate variable name", type=str)
parse.add_argument("-y", "--Y_var", help="Y coordinate variable name", type=str)
parse.add_argument("-k", "--Kind", help="Chart type", type=str, default='line')
parse.add_argument("-xs", "--X_scale", help="X-coordinate scale", type=int, default=10)
parse.add_argument("-w", "--Width", help="figure With", type=int, default=0)
parse.add_argument("-hi", "--High", help="figure High", type=int, default=0)

args = parse.parse_args()
filename = args.Filename
X_var = args.X_var
Y_var = args.Y_var
Kind = args.Kind
X_scale = args.X_scale
W = args.Width
H = args.High
figsize = (W, H)
if W == 0 or H == 0:
    figsize = None
df = pd.read_csv(filename, sep='\t', header=0)
df = df[(df['time'] > "2015-06-25 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
print df.head()

df['words_num'] = df['content'].apply(lambda x: len(x.replace("\n", "").replace(" ","").decode('utf-8')))
df['set_words_num'] = df['content'].apply(lambda x:len(set(x.replace("\n", "").replace(" ","").decode('utf-8'))))
sub_df = df[['content', 'fc', 'cc' ,'lc', 'words_num', 'set_words_num']]
sub_df.sort_values(by=[X_var], inplace=True)
plt.grid()
sub_df.plot(figsize=figsize, x=X_var, y=Y_var, xlim=(min(df[X_var]), max(df[X_var])), xticks=(range(min(df[Y_var]), max(df[Y_var]), X_scale)), kind=Kind)
plt.legend(loc='upper right') ## 图例的位置
plt.savefig(X_var+"_"+Y_var+"_"+Kind+'.png')
print X_var+"_"+Y_var+"_"+Kind+'.png'+" "+ "ploted successed!!!!"
