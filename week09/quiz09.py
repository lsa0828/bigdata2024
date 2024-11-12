'''
# Q9.1
from sklearn import linear_model
import pandas as pd

data = {'x' : [59, 49, 75, 54, 78, 56, 60, 82, 69, 83, 88, 94, 47, 65, 89, 70],
        'y' : [209, 180, 195, 192, 215, 197, 208, 189, 213, 201, 214, 212, 205, 186, 200, 204]}
data = pd.DataFrame(data)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(data["x"]), y = data["y"])

print('Y 절편 값: ', linear_regression.intercept_)
print('회귀 계수 값: ', linear_regression.coef_)

prediction = linear_regression.predict(X = pd.DataFrame(data["x"])) # y 예측값
residuals = data["y"] - prediction; # y 관측값 - y 예측값
SSE=sse= (residuals**2).sum()
SST = ((data["y"]-data["y"].mean())**2).sum() # (y 관측값 - y 평균)제곱 총합
R_squared = 1 - (SSE/SST)
print("R_squared: ", R_squared)
score = linear_regression.score(X = pd.DataFrame(data["x"]), y = data["y"])
print("score=>", score)

mydata = {'x' : [58],
        'y' : []}
prediction = linear_regression.predict(X = pd.DataFrame(mydata["x"]))
print("X가 58일 때 예측=>", prediction)
'''


# Q9.2
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

diabetes_data = datasets.load_diabetes()
X = pd.DataFrame(diabetes_data.data)
y = diabetes_data.target

print("데이터셋 크기: ", X.shape)
X.info()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, Y_train)

Y_predict = linear_regression.predict(X_test)

print('a value = ', linear_regression.intercept_)
print('b balue =', linear_regression.coef_)

residuals = Y_test-Y_predict
SSE = (residuals**2).sum(); SST = ((Y_test-Y_test.mean())**2).sum()
R_squared = 1 - (SSE/SST)

print('R_squared = ', R_squared)
print('score = ', linear_regression.score(X = pd.DataFrame(X_test), y = Y_test))
print('Mean_Squared_Error = ', mean_squared_error(Y_test, Y_predict))
print('RMSE = ', mean_squared_error(Y_test, Y_predict)**0.5)
