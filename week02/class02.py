def common_data(a, b):
    for i in a:
        if i in b:
            return "True"
    return "None"

print(common_data([1,2,3,4,5],[5,6,7,8,9]))
print(common_data([1,2,3,4,5],[6,7,8,9]))
'''
def test(lst, str):
    for l in lst:
        if l in str:
            return "True"
    return "False"

str1 = "https://www.w3resource.com/python-exercies/list/"
lst = ['.com','.edu','.tv']
print(test(lst,str1))
str1 = "https://www.w3resource.net"
lst = ['.com','.edu','.tv']
print(test(lst,str1))
'''

dic = {'name':'Hong', 'phone':'01012345678', 'birth':'0814'}
dic
print(dic)

dic.keys()
dic.values()
list(dic.values())
dic.items()
# dic.items(1)
dic['phone']
dic['pet'] = 'dog'
dic
# del dici[1]

k=range(5)
y=list(k)
type(y)

import urllib.request
urllib.request.Request('http://www.hanb.co.kr')

f=open('a.txt','w')
for i in range(1,5):
    data="%d번째 줄입니다 \n"%i
    f.write(data)
f.close()

f=open('a.txt','r')
lines=f.readlines()
for i in lines:
    print(i)
f.close()

import numpy as np
ar1 = np.array([1,2,3,4,5])
list1 = [1,2,3,4,5]
ar = np.array(list1)
ar2 = np.arange(1,11,2)
ar4 = np.array([1,2,3,4,5,6])
ar4 = np.array([1,2,3,4,5,6]).reshape(3,2)

import pandas as pd
pd.__version__
data1 = [10,20,30,40,50] # {'0':10, '1':20, '2':30, '3':40, '4':50} 인덱스가 숫자
data1
sr1 = pd.Series(data1)
sr1
data2 = ['1반','2반','3반','4반','5반']
sr2 = pd.Series(data2)
sr3 = pd.Series([101,102,103,104,105])
sr3.name
sr4 = pd.Series(['월','화','수','목','금'])
sr4
sr5 = pd.Series(data1, index = [1000,1001,1002,1003,1004])
sr6 = pd.Series(data1, index = data2)
sr6
sr7 = pd.Series(data2, index = data1)
sr7
sr8 = pd.Series(data2, index = sr4)
sr8
sr8.name = 'sr8'
sr1+sr3
sr4+sr2
