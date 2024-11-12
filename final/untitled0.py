'''
import pandas as pd
emp=pd.read_csv("emp.csv")
num=emp.groupby('job').empno.count()
print(num.sort_values(ascending=False))
'''
'''
import pandas as pd
mtcars = pd.read_csv("mtcars.csv")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
mtcars[['qsec']]=scaler.fit_transform(mtcars[['qsec']])

print(len(mtcars.loc[mtcars.qsec > 0.5]))
'''
'''
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score

y_test = [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
y_prediction = [1, 0, 1, 1, 1, 0, 1, 1, 1, 0]

confusion_matrix = confusion_matrix(y_test, y_prediction) # 오차 행렬([[TN, FP], [FN, TP]])
accuracy = accuracy_score(y_test, y_prediction) # 정확도
precision = precision_score(y_test, y_prediction) # 정밀도
recall = recall_score(y_test, y_prediction) # 재현율

print('정확도 : ', accuracy, '정밀도 : ', precision, '재현율', recall)

print('y_test=', y_test)
print('y_prediction=', y_prediction)
print('confusin matrix =>', confusion_matrix)

'''

import pandas as pd
smoke=pd.read_csv("smoke.csv")

smoke_d = smoke.groupby('smoker').charges.mean()

print(smoke_d['yes'] - smoke_d['no'])