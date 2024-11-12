import csv
import matplotlib.pyplot as plt
f = open('age.csv')
data = csv.reader(f)
next(data)
data = list(data)

name = '강남구 대치1동'
home = []
result_name = ''
result = 0

for row in data :
    if name in row[0]: #'강남구 대치1동' 이름이 포함된 행 찾기
        for i in row[3:]: #3번 인덱스 값부터 슬라이싱 0세~
            home.append(int(i)) #'강남구 대치1동' 데이터를 home에 저장
        hometotal=int(row[2])
for k in range(len(home)):
    home[k]=(home[k]/hometotal)
    
result_list=[]
for row in data : 
    away=[]
    for i in row[3:]: #3번 인덱스 값부터 슬라이싱 0세~
        away.append(int(i)) #입력 받은 지역의 데이터를 away에 저장
    awaytotal=int(row[2])
    for k in range(len(away)):
        away[k]=(away[k]/awaytotal)
    s=0
    for j in range(len(away)):
        s=s+(home[j]-away[j])**2
    result_list.append([row[0], away, s])
result_list.sort(key=lambda s: s[2])

plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)            
plt.rc('font', family ='Malgun Gothic')
plt.title('강남구 대치1동과 가장 비슷한 인구 구조를 가진 5곳')
plt.plot(home, label = name)
plt.plot(result_list[1][1], label = result_list[1][0])
plt.plot(result_list[2][1], label = result_list[2][0])
plt.plot(result_list[3][1], label = result_list[3][0])
plt.plot(result_list[3][1], label = result_list[4][0])
plt.plot(result_list[3][1], label = result_list[5][0])
plt.legend()
plt.show()