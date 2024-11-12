import pandas as pd
import numpy as np

data = {
        'apples': [3, 2, 0, 1],
        'oranges': [0, 3, 7, 2]
}

a = pd.DataFrame(data)
a = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
col1 = pd.Series([3, 2, 0, 1], name='apples')
col2 = pd.Series([3, 2, 0, 1], name='oranges', index=['A', 'B', 'C', 'D'])
col2 = pd.Series([3, 2, 0, 1], name='oranges')
col1.name # 'apples'
col1.values
col1.index

list(col1.values)
list(col2.index)

col3 = pd.concat([col1, col2], axis=1)
list(col3.columns)

df = pd.DataFrame([[89.1, 90.1, 'B'], [89.2, 90.2, 'A'], [89.3, 90.3, 'A'], [89.4, 90.5, 'C'], [89.5, 90.5, 'B']], 
                  index=['1반', '2반', '3반', '4반', '5반'], columns=['중간고사', '기말고사', '성적'])
df.중간고사
df.loc['1반':'2반']
df.iloc[1:3]
df.iloc[0:3]['중간고사']
df.loc['1반']

df[df.성적 == 'B']
df.loc[df.성적 == 'B'] # 위와 결과 같음
df[df.성적.isin(['B', 'C'])]
df[(df.성적 == 'A') & (df.기말고사 >= 90)]

list(df[df.성적 == 'B'].index)

list(df[df.성적 == 'B'].중간고사)

df.describe()
df.중간고사.unique()
df.중간고사.mean()
df.중간고사.value_counts() # 값이 몇 개씩 있는지
df.중간고사.map(lambda p : p - df.중간고사.mean()) # 평균과의 차이

df.groupby('중간고사').중간고사.count()
df.groupby('중간고사').기말고사.min()
df.groupby(['중간고사']).중간고사.agg([len, min, max])

df.sort_values(by='중간고사')
df.sort_values(by='중간고사', ascending=False)
df.sort_index(ascending=False)

df.dtypes
df.중간고사.dtypes
df.loc['5반']=[10, 10, np.nan]
df[pd.isnull(df.성적)]

df.rename(columns={'성적': '등급'})
df.rename_axis("반이름", axis='rows')

import matplotlib.pyplot as plt
x = [2016, 2017, 2018, 2019, 2020]
y = [350, 410, 520, 695, 543]
plt.plot(x, y) # x축과 y축 데이터 지정하여 라인플롯 생성
plt.title('Annual sales')
plt.xlabel('years') 
plt.ylabel('sales')
plt.show()

y1 = [350, 410, 520, 695]
y2 = [200, 250, 385, 350]
x = range(len(y1))
# x축과 y축 데이터 지정하여 라인플롯 생성
plt.bar(x, y1, width = 0.7, color = "blue")
plt.bar(x, y2, width = 0.7, color = "red", bottom = y1)
plt.title('Quarterly sales')
plt.xlabel('Quarters')
plt.ylabel('sales')
xLabel = ['first', 'second', 'third', 'fourth']
plt.xticks(x, xLabel, fontsize = 10) # rotation=’75’
plt.legend(['chairs', 'desks']) # y1이 chairs, y2가 desks
plt.show()

range(0, 100)

round(6.14665, 2)

[0] * 4

np.arange(0., 5., 0.2)

np.array([1, 2, 3, 4])

np.random.choice(10, 6) # 0~10까지 6개. 배열
np.random.choice(10, 6, replace=False) # 중복 금지

with open('./%s_%s_%d_%s.json' % (name, ed, year, data), 'w', encoding='utf8')
	as outfile:
