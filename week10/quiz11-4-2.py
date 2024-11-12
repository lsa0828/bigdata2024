from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# n_estimators 값에 따른 정확도를 저장할 리스트
n_estimators_range = range(1, 201, 10)
accuracies = []

# 각 n_estimators 값에 대해 RandomForestClassifier 학습 및 테스트
for n in n_estimators_range:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 정확도가 가장 높은 n_estimators 값과 정확도
best_n = n_estimators_range[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)

# 결과 출력
print(f"n_estimators가 {best_n}일 때 정확도가 {best_accuracy:.2f}로 가장 높습니다.")

# 정확도 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracies, marker='o')
plt.title('n_estimators 값에 따른 정확도')
plt.xlabel('n_estimators')
plt.ylabel('정확도')
plt.grid(True)
plt.show()
