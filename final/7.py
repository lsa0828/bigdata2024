# 7번
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. 데이터 읽기
X=pd.read_csv("X.csv")
y=pd.read_csv("y.csv")

# 2. 모델 학습
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(x_train, y_train['target'])

# 3. 예측 및 평가
y_pred = model.predict(x_test)
f1_score = f1_score(y_test['target'], y_pred)
print(f1_score)