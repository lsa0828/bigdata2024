import pandas as pd
import numpy as np
X_train = pd.read_csv("XX_train.csv")
Y_train = pd.read_csv("YY_train.csv")
X_test = pd.read_csv("X_test.csv")
#X_train.info()
#X_train['Warehouse_block'].unique() # ['B', 'F', 'C', 'A', 'D']
#X_train['Mode_of_Shipment'].unique() # ['Ship', 'Flight', 'Road']
#X_train['Product_importance'].unique() # ['high', 'medium', 'low']
#X_train['Gender'].unique() # ['F', 'M']

# 0. 분석에 필요하지 않은 컬럼 제거
#ID = X_train.pop('ID')
#ID = Y_train.pop('ID')
#ID = X_test.pop('ID')
ID = X_train['ID']
X_train = X_train.drop(columns='ID')
Y_train = Y_train.drop(columns='ID')
X_test = X_test.drop(columns='ID')

# 1. train-test 검증 데이터 분리 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=34)

# 2. 라벨 인코딩 - 명목형 변수, LabelEncoding(), get_dummies(), replace
x_train = pd.get_dummies(x_train, drop_first=True, dtype=float) # color_1, color_2, color_3 중에서 color_1은 안 나옴
x_test = pd.get_dummies(x_test, drop_first=True, dtype=float)
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in x_train.columns:
    if x_train[column].dtype == 'object': # 숫자형 열은 라벨 인코딩 X
        x_train[column] = le.fit_transform(x_train[column])
        x_test[column] = le.fit_transform(x_test[column])
'''

# 3. minmaxscaling, Standardscaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
'''
for column in x_train.columns:
    if x_train[column].dtype == 'int64':
        x_train[[column]] = scaler.fit_transform(x_train[[column]])
        x_test[[column]] = scaler.transform(x_test[[column]])
'''
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
'''

# 4. 모델 생성1(RandomForest)
'''
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression() # 0.6417560518140789
'''
'''
from sklearn.linear_model import LinearRegression # 보류
model1 = LinearRegression() # 0.7411560656681107
'''

from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=5, random_state=10) # 0.7417941504015004

'''
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, random_state=42) # 0.694185569427349
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

# y_pred = model.predict_proba(x_test)[:, 1]

# 6. CSV 파일 생성
#output = pd.DataFrame({'ID': ID, 'Target': y_test, 'Predicted': y_pred1})
#output.to_csv('predictions.csv', index=False)


# 모델 만들기
# - encoding, scaling하고 split하는 방법
# - split하고 encoding, scaling을 각각 두번 하는 방법



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


