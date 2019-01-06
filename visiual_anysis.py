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
parse.add_argument("-f","--filename", help="need to plot filename", type=str)
parse.add_argument("-x", "--X_var", help="X coordinate variable name", type=str)
parse.add_argument("-y", "--Y_var", help="Y coordinate variable name", type=str)
args = parse.parse_args()
filename = args.filename
X_var = args.X_var
Y_var = args.Y_var
df = pd.read_csv(filename, sep='\t', header=0)
df = df[(df['time'] > "2015-06-25 00:00:00") & (df['time'] < "2015-07-01 00:00:00")]
print df.head()

df['words_num'] = df['content'].apply(lambda x: len(x.replace("\n", "").replace(" ","").decode('utf-8')))
df['set_words_num'] = df['content'].apply(lambda x:len(set(x.replace("\n", "").replace(" ","").decode('utf-8'))))
sub_df = df[['content', 'fc', 'cc' ,'lc', 'words_num', 'set_words_num']]
sub_df.sort_values(by=[X_var], inplace=True)
#print sub_df[sub_df['fc']>6000]
#plt.figure(figsize=(20,15),facecolor='w')##设置大小和背景颜色
## 我们下面绘制的四幅图都是用的上面同一个plt，故下面四条线都在一张图中显示，如果想在不同图中显示，只需要在plt.plot之前重新定义一个figsize即可。
#plt.plot(sub_df['words_num'].values, sub_df['fc'].values, 'r-', label='fc', linewidth=2)
#plt.plot(sub_df['words_num'], sub_df['cc'], 'g-', label='cc', linewidth=2)
#plt.plot(sub_df['words_num'], sub_df['lc'], 'b-', label='lc', linewidth=2)
plt.grid()
sub_df.plot(x=X_var, y=Y_var)
plt.legend(loc='upper right') ## 图例的位置
plt.savefig(X_var+"_"+Y_var+'.png')
print "plot successed!!!!"
