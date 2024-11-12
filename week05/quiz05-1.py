import csv
import matplotlib.pyplot as plt
f1 = open('201803subway.csv', encoding='cp949')
f2 = open('202003subway.csv', encoding='cp949')
f3 = open('202303subway.csv', encoding='cp949')
data1 = csv.reader(f1)
data2 = csv.reader(f2)
data3 = csv.reader(f3)
next(data1)
next(data1)
next(data2)
next(data2)
next(data3)
next(data3)

s1_in = [0] * 24
s1_out = [0] * 24
s2_in = [0] * 24
s2_out = [0] * 24
s3_in = [0] * 24
s3_out = [0] * 24
for row in data1 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s1_in[i] += row[4 + i * 2] # 시간대별 승차인원
        s1_out[i] += row[5 + i * 2] # 시간대별 하차인원
for row in data2 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s2_in[i] += row[4 + i * 2] # 시간대별 승차인원
        s2_out[i] += row[5 + i * 2] # 시간대별 하차인원
for row in data3 :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s3_in[i] += row[4 + i * 2] # 시간대별 승차인원
        s3_out[i] += row[5 + i * 2] # 시간대별 하차인원

plt.figure(dpi = 300)
plt.rc('font', family = 'Malgun Gothic')
plt.title('지하철 시간대별 승하차 인원 추이')
plt.plot(s1_in, label = '201803승차')
plt.plot(s1_out, label = '201803하차')
plt.plot(s2_in, label = '202003승차')
plt.plot(s2_out, label = '202003하차')
plt.plot(s3_in, label = '202303승차')
plt.plot(s3_out, label = '202303하차')
plt.legend()
plt.xticks(range(24), range(4,28))
plt.show()