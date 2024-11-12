import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mtcars_train=pd.read_csv('mtcartrain.csv')
mtcars_test=pd.read_csv('mtcartest.csv')

m_train, m_test = train_test_split(mtcars_train, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X=pd.DataFrame(m_train[['drat', 'wt', 'gear', 'carb']]), y=m_train['mpg'])

m_predict = model.predict(pd.DataFrame(m_test[['drat', 'wt', 'gear', 'carb']]))

mse = mean_squared_error(m_test['mpg'], m_predict)
print("mse => ", round(mse, 3))

pred = model.predict(pd.DataFrame(mtcars_test[['drat', 'wt', 'gear', 'carb']]))
print("mtcarstest 연비 예측 =>", pred[0:5])
