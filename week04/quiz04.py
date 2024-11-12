# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:56:49 2024

@author: 이수아
"""
#[Q4.1]
import csv
import numpy as np
import matplotlib.pyplot as plt

f1 = open('201402pop.csv')
f2 = open('202402pop.csv')
data1 = csv.reader(f1)
data2 = csv.reader(f2)
next(data1)
next(data2)
result = []
bars = []
        
for (row1, row2) in zip(data1, data2):
    bars.append(row1[0])
    p2024 = float(row2[1].replace(',', ''))
    p2014 = float(row1[1].replace(',', ''))
    result.append(p2024-p2014)
            
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.bar(bars, result)
plt.xticks(np.arange(len(bars)), bars, rotation='vertical')
plt.savefig("quiz_img1.png", bbox_inches='tight')


#[Q4.2]
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('201402pop.csv', encoding='cp949')
df2 = pd.read_csv('202402pop.csv', encoding='cp949')
result = []
bars = []
        
for i in range(0, len(df1)):
    bars.append(df1.iloc[i, 0])
    p2024 = float(df2.iloc[i, 1].replace(',', ''))
    p2014 = float(df1.iloc[i, 1].replace(',', ''))
    result.append(p2024-p2014)
    
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.bar(bars, result)
plt.xticks(np.arange(len(bars)), bars, rotation='vertical')
plt.savefig("quiz_img2.png", bbox_inches='tight')