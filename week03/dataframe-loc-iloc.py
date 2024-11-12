import numpy as np
import pandas as pd

# creating
df = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])
df['1반']['중간고사'] # error
df['중간고사']['1반'] # column이 앞에 있어야 함
df['중간고사'][0] # 키를 위치로 취급하는 것은 권장되지 않음(슬라이싱 하면 괜찮음)
df.loc['1반']['중간고사'] # index가 앞에 있으려면 .loc 붙여야 함
type(df)

# indexing-selection-assigning
df['중간고사']
type(df['중간고사']) # pandas.core.series.Series
df['중간고사'][0:2] # 슬라이싱 하면 괜찮음
type(df['중간고사'][0]) # numpy.float64
type(df['중간고사'][0:2]) # pandas.core.series.Series
df['중간고사']['1반':'2반']
type(df['중간고사']['1반':'2반']) # pandas.core.series.Series

# loc
df.loc['1반'] # object
type(df.loc['1반']) # pandas.core.series.Series
df.loc[:, '중간고사'] # float64
type(df.loc[:, '중간고사']) # pandas.core.series.Series
df.loc['1반':'2반']['중간고사'] # float64
df.loc['1반', '중간고사']
type(df.loc['1반', '중간고사']) # numpy.float64
df.loc['1반'][0]

df.iloc[0] # object
type(df.iloc[0]) # pandas.core.series.Series
df.iloc[0]['중간고사']
type(df.iloc[0]['중간고사']) # numpy.float64

df.loc[df.성적 == 'B']; x=df[df.성적 == 'B'];
cond = df.성적 == 'B'
df[cond]
df.loc[cond]

df.loc[(df.성적 == 'A') & (df.중간고사 >= 90)] # 가로 빼면 식이 바뀔 수 있음(가로 안이 우선순위로 계산)
cond1 = (df.성적 == 'A')
cond2 = (df.중간고사 >= 90)
df[cond1 & cond2]

df.loc[df.성적.isin(['B', 'C'])]

## summary function and maps
df.describe() # 통계값
df.info() # 전체적인 타입
df.중간고사.describe()
df.head(1)
df.중간고사.unique() # 값 표시(중복값은 한개만)
df.성적.unique()
df.중간고사.mean() # 평균
df.describe().loc['mean', '중간고사']
df.중간고사.value_counts() # 중복 개수
df_mean = df.중간고사.mean()
df.중간고사.map(lambda p: p - df_mean) # 값 - 평균

## grouping and sorting
df.groupby('중간고사').중간고사.count()
df.groupby('성적').count()
df.groupby('성적').count()['중간고사']
df.groupby('성적').sum()
df.groupby('중간고사').중간고사.min()
df.groupby(['중간고사']).중간고사.agg([len, min, max]) # 여러 가지 구하고 싶을 때
df.sort_values(by='중간고사')
df.sort_values(by='중간고사', ascending=False)
df.sort_index(ascending=False)

# data types and missing values
df.dtypes
df.info()
df.중간고사.dtypes
df.loc['6반']=[10, 10, np.nan]
df['7반']=[10, 10, np.nan] # error
df[pd.isnull(df.성적)]

# renaming and combining
df.rename(columns={'성적': '등급'}) # '성적'을 '등급'으로 바꾸는 건데 실제로는 바뀌지 않음
df = df.rename(columns={'성적': '등급'}) # 바꾸려면 치환해야 함
df.rename_axis("반이름", axis='rows')

df1 = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])

df0=pd.concat([df, df1]) # df와 df1을 아래로 붙여서 df0
df3=pd.concat([df, df1], axis=1) # df와 df1을 옆에 붙여서 df3

