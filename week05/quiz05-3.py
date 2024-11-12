import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('age.csv', encoding='cp949')
name = '강남구 대치1동'
home = []
result_name = ''
result = 0

for i in range(1, len(df)):
    if name in df.iloc[i, 0]:
        hometotal=int(df.iloc[i, 2])
        for j in df.iloc[i, 3:]:
            home.append(int(j))
for k in range(len(home)):
    home[k]=(home[k]/hometotal)

result_list=[]
for i in range(1, len(df)):
    away=[]
    awaytotal=int(df.iloc[i, 2])
    for j in df.iloc[i, 3:]:
        away.append(int(j))
    for k in range(len(away)):
        away[k]=(away[k]/awaytotal)
    s=0
    for l in range(len(away)):
        s=s+(home[l]-away[l])**2
    result_list.append([df.iloc[i, 0], away, s])
result_list.sort(key=lambda s: s[2])

plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)            
plt.rc('font', family ='Malgun Gothic')
plt.title('강남구 대치1동과 가장 비슷한 인구 구조를 가진 5곳(pandas 사용)')
plt.plot(home, label = name)
plt.plot(result_list[1][1], label = result_list[1][0])
plt.plot(result_list[2][1], label = result_list[2][0])
plt.plot(result_list[3][1], label = result_list[3][0])
plt.plot(result_list[3][1], label = result_list[4][0])
plt.plot(result_list[3][1], label = result_list[5][0])
plt.legend()
plt.show()