import csv
f = open('seoul.csv', encoding='cp949')
data = csv.reader(f)
header = next(data)
for row in data:
    print(row)
f.close()

row[-1] = float(row[-1]) # 실수로 변환 ~ int(a)는 정수로 변환

import matplotlib.pyplot as plt
# 선 그래프(꺾은 선 그래프)
plt.plot([10, 20, 30, 40]) # 데이터셋(y축)
plt.plot([1, 2, 3, 4], [12, 43, 25, 15]) # x축 데이터셋, y축 데이터셋(1에 12, 2에 43)
plt.plot([10, 20, 30, 40], color='skyblue', label='aaa') # 색깔, 범례 지정
plt.legend() # 범례 그래프에 나타냄

plt.plot([10, 20, 30, 40], 'r.') # 데이터를 빨간색 원형으로 나타냄
plt.plot([10, 20, 30, 40], 'g^') # 데이터를 초록색 삼각형으로 나타냄

result = []
result.append(a)

s.split() # 문자열 분리해서 배열로 나타냄 ['hello', 'python']
s.split()[0] # hello

# 히스토그램
plt.hist([1,1,2,3,4,5,6,7,8,10]) # 1이 몇 개인지 나타냄
plt.hist(dice, bins=6, color='r')

import random
random.randint(1, 6)

# 상자 그림
plt.boxplot(result) # 1이 몇 개인지 나타냄

#print('신도림' in '서울특별시 구로구 신도링동') # True
#if '신도림' in row[-1]: # '신도림'이 포함되면 True

# 격자 무늬 스타일 지정
plt.style.use('ggplot')
plt.plot(result)

name = input('찾고 싶은 지역의 이름은 : ')

plt.rcParams['axes.unicode_minus'] = False # 한글 폰트 사용 시 마이너스 부호 표현

plt.barh(range(101), result) # 수평 막대그래프(x축과 y축 바뀜)

f.reverse() # 배열 순서 뒤집기

# 두 데이터로 항아리 모양 그래프 만들려면 한 데이터에는 전체적으로 -(마이너스) 붙여서 반대로 모양이 나타나게 하기(plt.barh)

len(m) # 데이터 행 수(배열 크기)

# 파이 그래프(원 그래프)
plt.pie([10, 20])
plt.pid(result, labels=['A', 'B', 'C'])
plt.axis('equal')

# 산점도 표현(빨간색 원으로 데이터 표시)
plt.scatter([1, 2, 3, 4], [10, 30, 20, 40])
#plt..scatter([1, 2, 3, 4], [10, 30, 20, 40], s=[100, 200, 250, 300]) # 버블 차트로 표현
#					c=['red', 'blue', 'green', 'gold']
#					c=range(4) # 컬러바 추가-1
plt.colorbar() # 컬러바 추가-2

round(rate, 2) # 두 자리 소수점, 반올림
[0] * 4 # [0, 0, 0, 0]

row[4:] = map(int, row[4:]) # 문자열 -> 정수로 타입 바꿈

result.sort() # 오름차순으로 정렬
plt.bar(range(len(result)), result)

sum(row[10:15:2])

np.array([1, 2, 3, 4]) # array([1, 2, 3, 4])
np.sqrt(2) # 루트 2(제곱근)
np.arange(0., 5., 0.2) # 0부터 4.8까지 0.2씩 더한 array([~~])
np.random.choice(10, 6) # 0~10까지 6개. 배열
np.random.choice(10, 6, replace=False) # 중복 금지

#with open('./%s_%s_%d_%s.json' % (name, ed, year, data), 'w', encoding='utf8')
#	as outfile: