import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)

dir(pd)
pd.read_csv.__doc__
print(pd.read_csv.__doc__)
print(pd.DataFrame.head.__doc__)

print(data.head())
print(data.shape)
print(type(data))
print(data.columns)
print(data.describe())
print(data['hp'].describe())
print(data['gear'].unique())
print(data['cyl'].unique())

print(data.info())

print(data.corr())
X=data.drop(columns='mpg') # mpg(종속변수(target))만 빼고 나머지 X에 저장
Y=data['mpg'] # mpg 가짐, Type: Series
