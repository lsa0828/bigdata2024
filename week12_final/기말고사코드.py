#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
X_train = pd.read_csv("XX_train.csv")
Y_train = pd.read_csv("YY_train.csv")
X_test = pd.read_csv("X_test.csv")
#X_train.info()
#X_train.head()
#X_train.head().T
#Y_train.head().T
#X_train['Warehouse_block'].unique()
#X_train['Mode_of_Shipment'].unique()
#X_train['Product_importance'].unique()
#X_train['Gender'].unique()

# 0. 분석에 필요하지 않은 컬럼 제거
#ID = X_train.pop('ID')
#ID = Y_train.pop('ID')
ID = X_train['ID']
X_train = X_train.drop(columns='ID')
Y_train = Y_train.drop(columns='ID')
X_test = X_test.drop(columns='ID')

# 1. 라벨 인코딩 - 명목형 변수, LabelEncoding(), get_dummies(), replace
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in X_train.columns:
    if X_train[column].dtype == 'object': # 숫자형 열은 라벨 인코딩 X
        X_train[column] = le.fit_transform(X_train[column])
'''

# 2. minmaxscaling, Standardscaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # 주의
'''
for column in X_train.columns:
    if X_train[column].dtype == 'int64':
        X_train[[column]] = scaler.fit_transform(X_train[[column]])
'''
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
'''

# 3. train-test 검증 데이터 분리 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=34)

# 4. 모델 생성1(RandomForest)
'''
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression() # 0.6417560518140789
'''
'''
from sklearn.linear_model import LinearRegression # 보류(종속변수가 연속적인 값을 가질 때 사용)
model1 = LinearRegression() # 0.7410548246665707
'''

from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=5, random_state=10) # 0.7417941504015004

'''
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=10) # 0.6913281753707285
'''
'''
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier() # 0.7040472422484028
'''
'''
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=200, max_features=3, oob_score=True) # 0.6893646327882305
'''
model1.fit(x_train, y_train['Target'])

# 5. 모델 성능 평가
from sklearn.metrics import roc_auc_score
y_pred1=model1.predict(x_test)
print('RF',roc_auc_score(y_test, y_pred1))

'''
model1.fit(X_train, Y_train['Target'])
from sklearn.metrics import roc_auc_score
y_pred=model1.predict(X_test) # output
output = pd.DataFrame({'ID': ID, 'Predicted': y_pred})
output.to_csv('predictions.csv', index=False)
'''

# y_pred = model.predict_proba(x_test)[:, 1]

# 6. CSV 파일 생성
#output = pd.DataFrame({'ID': ID, 'Target': y_test, 'Predicted': y_pred1})
#output.to_csv('predictions.csv', index=False)
#output.to_csv('predictions.csv', header=False)


# 모델 만들기
# - encoding, scaling하고 split하는 방법 : 데이터 누수 위험. 특히 스케일링 과정에서 통계값이 테스트 세트에 영향 미칠 수 있음
# - split하고 encoding, scaling을 각각 두번 하는 방법 : 데이터 누수 방지. 코드 복잡해질 수 있음 (추천)



'''
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
confusion_matrix(y_test, y_predict) # 오차 행렬 / [[TN, FP], [FN, TP]]

acccuracy = accuracy_score(y_test, y_predict); # 167/171
precision = precision_score(y_test, y_predict); # 107/110
recall = recall_score(y_test, y_predict); # 107/108
f1 = f1_score(y_test, y_predict)
roc_auc = roc_auc_score(y_test, y_predict)

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f},  F1: {3:.3f}'.format(acccuracy,precision,recall,f1))
print('ROC_AUC: {0:.3f}'.format(roc_auc))
'''


