# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:07:35 2023
@author: Park
"""
# OneHotEncoder ##############################################################################
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
df = pd.DataFrame([
    [2,1,'male','A',3],
    [3,2,'female','C',5],
    [3,4,'male','B',7],
    [5,5,'female','A',10],
    [7,5,'female','B',12],
    [2,5,'male','A',7],
    [9,2,'male','C',13]
], columns=['hours', 'attendance', 'sex', 'cate', 'score'])
df.info()

ohe = OneHotEncoder(sparse=False)
ohe.fit(df[['sex']]) # Series encoding
encoding1=ohe.transform(df[['sex']]) # array of float64
print(ohe.transform(df[['sex']]))

array1=np.array(df[['sex']]) 
ohe = OneHotEncoder(sparse=False)
ohe.fit(array1) # array encoding
encoding1=ohe.transform(df[['sex']]) # array of float64
print(ohe.transform(df[['sex']]))

ohe = OneHotEncoder(sparse=False)
ohe.fit(df) # dataframe encoding
encoding1=ohe.transform(df) # array of float64; dir(ohe); ohe.categories_; help(ohe.fit)
print(ohe.transform(df[['sex']]))

ohe.fit(df[['cate']])
cateencodeing=ohe.transform(df[['cate']])
print(ohe.transform(df[['cate']]))

df = pd.DataFrame([
    [2,1,'A',3],
    [3,2,'C',5],
    [3,4,'B',7],
    [5,5,'A',10],
    [7,5,'B',12],
    [2,5,'A',7],
    [9,2,'C',13]
], columns=['hours', 'attendance', 'cate', 'score'])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit(df[['cate']])
print(ohe.transform(df[['cate']]))

df_cate = pd.DataFrame(ohe.transform(df[['cate']]), columns=['cate0', 'cate1', 'cate2'])
print(df_cate)

df = pd.concat([df[['hours', 'attendance']], df_cate, df[['score']]], axis=1)
# axis=0(기본, 행row 기준으로 결합) axis=1(열column 기준으로 결합)
print(df)
# df['hours']와 df[['hours']]의 차이점 : column 이름 나오는지
x_data = df[['hours', 'attendance', 'cate0', 'cate1', 'cate2']]
y_data = df['score']
print(x_data)
print(y_data)

# LabelEncoder ##############################################################################
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
#LabelEncoder()
le.classes_
# array([1, 2, 6])
le.transform([1, 1, 2, 6])
# array([0, 0, 1, 2]...)
le.inverse_transform([0, 0, 1, 2])
# array([1, 1, 2, 6])

# non-numerical labels (as long as they are hashable and comparable) to numerical labels.
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
#LabelEncoder()
list(le.classes_)
# ['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"])
# array([2, 2, 1]...)
list(le.inverse_transform([2, 2, 1]))
# ['tokyo', 'tokyo', 'paris']

# check
le.fit(['A', 'A', 'C', 'E'])
list(le.classes_)   # ['A', 'C', 'E']
list(le.transform(['C', 'C', 'A'])) # [1, 1, 0]
list(le.inverse_transform([2, 2, 1]))   # ['E', 'E', 'C']


###############################################################################
import pandas as pd
import numpy as np
s = pd.Series(list('abca')); s
ss=pd.get_dummies(s); ss

s1 = ['a', 'b', np.nan]
ss1=pd.get_dummies(s1); ss1
ss2=pd.get_dummies(s1, dummy_na=True); ss2
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],'C': [1, 2, 3]}); df
ss3=pd.get_dummies(df, prefix=['col1', 'col2']); ss3
ss4=pd.get_dummies(pd.Series(list('abcaa'))); ss4
ss5=pd.get_dummies(pd.Series(list('abcaa')), drop_first=True); ss5
ss6=pd.get_dummies(pd.Series(list('abc')), dtype=float); ss6

p1=pd.DataFrame(['a', 'b', np.nan]); p1
pp1=pd.get_dummies(p1); pp1

p1=pd.DataFrame([['a', 'b', np.nan], ['a', np.nan, 'a']]); p1
pp1=pd.get_dummies(p1); pp1
