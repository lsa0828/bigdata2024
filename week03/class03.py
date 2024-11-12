# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:06:08 2024

@author: 이수아
"""

import pandas as pd

data = {
'apples': [3, 2, 0, 1], 
'oranges': [0, 3, 7, 2]
}

purchases = pd.DataFrame(data)
purchases
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])
purchases
col1=pd.Series([3, 2, 0, 1], name='apples')
col2=pd.Series([3, 2, 0, 1], name='oranges', index=['June', 'Robert', 'Lily', 'David'])
col1
col1.name
col1.values
col1.index
purchases2 = pd.DataFrame(col1)
purchases2
purchases3 = pd.DataFrame(col2)
purchases3
col2=pd.Series([3, 2, 0, 1], name='oranges')
purchases4= pd.concat([col1, col2], axis=1)
purchases4
purchases4.index=['June', 'Robert', 'Lily', 'David']
purchases4
#purchases4.name
purchases4.index
purchases4.columns

data1 = [10, 20, 30, 40, 50]
data2 = ['1반', '2반', '3반', '4반', '5반']

src3 = pd.Series([101, 102, 103, 104, 105])
src3

df2 = pd.DataFrame([[89.2, 92.5, 90.8], [92.8, 89.9, 95.2]], 
                   index=['중간고사', '기말고사'], columns=data2[0:3])
df2

import pandas as pd
df = pd.DataFrame([[60, 61, 62], [70, 71, 72], [80, 81, 82], [90, 91, 92]],
index = ['1반', '2반', '3반','4반'], columns = ['퀴즈1', '퀴즈2', '퀴즈3'])
#df : 열 선택
df.퀴즈1
df['퀴즈1']
df['퀴즈1'][2]
#df.loc : 행 선택, 행열선택
df.loc['2반']
df.loc['2반', '퀴즈1']
df.loc['2반':'4반', '퀴즈1'] # type(df.loc[＇2반＇:＇4반＇, ＇퀴즈1＇])
df.loc['2반':'4반', '퀴즈1':'퀴즈3'] # type(df.loc[＇2반＇:＇4반＇, ＇퀴즈1＇:＇퀴즈2＇])
#df.iloc : 행 선택, 행열선택
df.iloc[2]
df.iloc[2, 1]
df.iloc[2:4, 0]
df.iloc[2:4, 0:2]
df.iloc[2:4, 0:1]