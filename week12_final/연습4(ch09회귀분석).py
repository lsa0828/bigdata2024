# data = {'x': [4,6,7,7,8,10], 'y': [2,4,6,8,7,9]}
# x값 평균 = mean(x) = 7 = m(x)
# y값 평균 = mean(y) = 6 = m(y)
# SST = (y-m(y))^2 = 34 = 모든 분산 구해서 더함 = 실제값의 분산
# y=1.2x-2.4라는 선형 회귀식으로 실제 x값으로 y값을 예측함
# 예측한 y 값으로 모든 분산(평균은 실제 평균 사용) 구해서 더함 = SSR = 예측값의 분산
# MA(절대)E = |실제 y - 예측 y| 구해서 평균 구함
# MS(sqrt)E = SSE = (실제 y - 예측 y)^2 구해서 평균 구함. RMSE = np.sqrt(MSE)
# R2 score(결정계수 = R-squared = R 제곱, 설명력) = SSR/SST = 1-SSE/SST

# y로 행함. y_test, y_pred
# SST = (실제 - 평균)^2 모두 더함
# SSR = (예측 - 평균)^2 모두 더함
# SSE = (실제 - 예측)^2 모두 더함
# SSE로 평균 내면 MSE
# |실제 - 예측| 평균 내면 MAE
# R2 score(결정계수) = SSR/SST = 1 - SSE/SST
# SSR = SST - SSE

#선형회귀식 구하는 코드
data = {'x': [156, 160, 169, 167, 154, 163], 'y': [51, 54, 62, 61, 49, 55]}
import sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X=pd.DataFrame(data['x']), y=data['y']) # 독립 변수는 x, 종속변수는 y
print('절편 = ', model.intercept_)
print('기울기 = ', model.coef_)
# y = coef * x + intercept

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
r2score = r2_score(Y_test, Y_predict)

# <질문>
# 다중 회귀분석의 경우 식이 의미가 있는지 어떻게 판단하는가? F 통계량을 사용하여 p-값이 0.05보다 작으면 유의하다고 판단
# 모형이 얼마나 설명력을 갖는가? 결정계수(R^2, R2 score) 확인
# 독립 변수들이 많을 경우 독립 변수가 의미가 있는지 판단은 어떻게 하는가? 각 변수의 p-값이 0.05보다 작으면 유의하다고 판단

# 회귀계수 확인하여 독립 변수가 종속변수에 미치는 영향 분석
coef = pd.Series(data = np.round(model.coef_, 2), index = X.columns)
coef.sort_values(ascending=False)

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2) # 2행 3열

# 선형 회귀 : 실제값과 예측값의 오차에 기반한 지표. 종속변수가 연속형(숫자)
# 로지스틱 회귀 : 선형 회귀와 달리 S자 함수 사용하여 참(True, 1)과 거짓(False, 0) 분류. 종속변수가 범주형이며 0, 1 값을 가짐
# 시그모이드 함수 : 로지스틱 회귀에 사용하는 S자 함수. x값이 커지면 y 값은 1에 근사하고 x값 작아지면 y값은 0에 근사하게 되어 S자 형태의 그래프가 됨. 두 개의 값을 분류하는 이진 분류에 많이 사용
# 오차 행렬(혼동행렬, Confusion Matrix)
# TN(Negative가 참), TP(Positive가 참), FN(Negative가 거짓), FP(Positive가 거짓)

# 정확도(accuray) : (TN + TP)/전체
# 오류율 = (FP + FN)/전체
# 정밀도(precision) : TP/(FP+TP) : P라고 예측한 것 중 실제 P
# 재현율(recall, 민감도, TPR) : TP/(FN+TP) : 실제 P인 것 중 P라고 예측
# F1 스코어 : 2*(정밀도*재현율)/(정밀도+재현율)
# FPR(Fall-out) : FP/(FP+TN)
# 특이도(specificity) = 1-FPR = TN/(TN+FP)
# ROC 기반 AUC 스코어 : roc_auc_score 함수 제공. AUC 값이 커지려면 FPR이 작을 때 TPR이 커야 함

# 로지스틱 회귀 분석에 쓰일 데이터를 정규분포(평균이 0, 분산이 1) 형태로 맞춤
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model = scaler.fit_transform(data)

# 로지스틱 회귀 분석 모델 구축
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 모델 성능 확인
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
confusion_matrix(y_test, y_pred) # 오차 행렬([[TN, FP], [FN, TP]])
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Decision Tree(의사 결정 트리)
# -범주형 목표변수(분류 트리(Classification Tree))
# -연속형 목표변수(회귀 트리(Regression Tree))
# 지니 지수(Gini index) : 데이터 집합의 불순도 측정. 0~1 사이의 값. 지니 지수가 작을수록 잘 분류된 것. class=N과 Y는 목표변수의 No(0)와 Yes(1)

data.Humidity.replace('High', 6)
x = np.array(pd.DataFrame(data, columns = ['A', 'B', 'C', 'D']))
y = np.array(pd.DataFrame(data, columns = ['E']))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

classification_report(y_test, y_pred) # precision, recall, f1-score, support 다 나옴

# Random Forest : 여러 개의 Decision Tree를 결합함으로 단일 Decision Tree의 결점 극복. Over-fitting 문제 적음. 구현 간단. 병렬 계산 간편. tree의 개수 많을수록 좋은 건 아님. 분류 모델에서 training data와 test data의 선택은 매우 중요함
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)
model.score(x_test, y_test) # 정확도 측정