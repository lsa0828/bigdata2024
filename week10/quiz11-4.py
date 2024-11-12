from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#loading the iris dataset
iris = load_iris()

# to excel... by Uchang
import pandas as pd 
df = pd.DataFrame(data=iris['data'], columns = iris['feature_names'])
df.to_excel('iris.xlsx', index=False)

#training data 설정 
x_train = iris.data[:-30]
y_train = iris.target[:-30]
#test data 설정
x_test = iris.data[-30:] # test feature data  
y_test = iris.target[-30:] # test target data

#RandomForestClassifier libary를 import
from sklearn.ensemble import RandomForestClassifier # RandomForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
x = iris.data
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print (y_test)
print (Y_test)


n_estimators_range = range(1, 11)
for n in n_estimators_range:
    clf = RandomForestClassifier(n_estimators=n) # Random Forest
    clf.fit(X_train, Y_train)
    prediction_1 = clf.predict(X_test)
    print (n, " Accuracy is : ", accuracy_score(prediction_1, Y_test))


# Initialize the model
clf_2 = RandomForestClassifier(n_estimators=200, # Number of trees
                               max_features=4,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*
clf_2.fit(X_train, Y_train)
prediction_2 = clf_2.predict(X_test)
print (prediction_2 == Y_test)
print ("Accuracy is : ",accuracy_score(prediction_2, Y_test))
print ("=======================================================")
print (classification_report(prediction_2, Y_test))

for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
    print(feature, imp)
    
import graphviz
import os
#os.environ['PATH'] += os.pathsep + 'c:\programdata\anaconda3\lib\site-packages'
#os.environ['PATH'] += os.pathsep + 'C:\Program Files\Graphviz\bin'
estimator = clf_2.estimators_[5]
from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# 생성된 .dot 파일을 .png로 변환
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=50'])

# jupyter notebook에서 .png 직접 출력
from IPython.display import Image

# 다음 명령어는 따로 실행해본다.
Image(filename = 'decistion-tree.png')


myX_test=np.array([[5.6,2.9,3.6,1.3]])
myprediction = clf_2.predict(myX_test)
print(myprediction)
