# 연속변수(연속적인 값), 이산변수(범주변수, ex) 성별(남자, 여자))
# 척도(scale) : 측정된 변수의 값을 표현하는 수준
# 명목척도 : 측정값이 같고 다름을 말할 수 있음. 측정값들 사이에 순서 없음 ex) 성별
# 순위척도, 서열척도 : 측정값들 사이에 순서 있음. 사칙연산 불가능 ex) 직급
# 구간척도, 등간척도 : 측정값들 사이에 순서 있고 간격 일정. 덧셈, 뺄셈 가능 ex) 온도
# 비율척도 : 구간척도 + 절대영점. 사칙연산 모두 가능 ex) 길이
# -> 척도에 따라 적용가능한 통계분석 방법 다름. 숫자로 표현된 경우라 하더라도 사칙	연산이 가능하지는 않음. 그 숫자의 의미(=척도)를 이해해야 함

# 도수분포표 : 데이터를 구간으로 나누어 각 구간의 빈도 나타낸 표
# 히스토그램 : 도수분포표를 그래프로 그린 것

# 중심경향치 : 자료의 중심을 나타내는 숫자. 무엇을 중심으로 모였는가 ex) 평균, 중간값
x = [100, 100, 200, 400, 1700]
np.mean(x)
np.median(x) # 중간값
from scipy.stats import mode
mode(x) # 최빈값

# 변산성 : 자료가 흩어져 있는 정도
# 변산성 측정치 : 모여있는 정도 혹은 흩어져 있는 정도
np.min(x)
np.max(x)
np.var(x) # 분산(평균에서 데이터가 벗어난 정도)
np.std(x) # 표준편차(=np.sqrt(np.var(x)))
np.quantile(x, .25) # 사분위간 범위(제1사분위수: 25% 지점, 제2사분위수: 50% 지점)
# 대부분 자료는 중심경향치 주변에 몰려있음. 변산성 측정치 기준으로 벗어난 정도 파악
# 평균에서 벗어난 정도를 판단할 때는 표준편차 사용
# 중간값에서 벗어난 정도 판단할 때는 IQR 사용
# 중심경향치에서 크게 벗어났다면 이상점으로 의심할 수 있음

# 공분산 : 두 변수가 함께 변화하는 정도 나타내는 지표
# 상관계수 : 두 변수가 함께 변화하는 정도를 -1~1 범위의 수로 나타낸 것
# +인 경우 : 두 변수가 같은 방향으로 변화
# -인 경우 : 두 변수가 반대 방향으로 변화
# 0인 경우 : 두 변수가 독립적인 것. 한 변수의 변화로 다른 변수의 변화 예측 못 함
# 상관관계가 있다고 반드시 인과관계가 있는 것은 아님. 두 변수의 관계가 선형적(=직선)인지 확인할 것. 산점도(plt.scatter) 그려서 확인

# 회귀 분석 : 독립 변수 x로 종속변수 y 예측
# 독립 변수 : 변수의 변화 원인이 모형 밖에 있음
# 종속변수 : 변수의 변화 원인이 모형 안에 있음
# 선형 회귀 분석 : 독립 변수와 종속변수 사이에 직선적인 형태의 관계가 있다고 가정. 독립 변수가 일정하게 증가하면 종속변수도 그에 비례해서 증가 또는 감소
# ex) 식사량이 증가하면 체중이 얼마나 증가하는지 알 수 있다.
# y = wx + b	w : 회귀계수(coefficient), 직선의 기울기(slope)
#			b : y 절편(intercept), 독립 변수가 모두 0일 때 종속변수 y의 값
# 회귀 분석으로 모형적합도(모형이 데이터에 얼마나 잘 맞는가), 회귀계수(독립 변수의 변화가 종속변수를 얼마나 변화시키는가)를 알 수 있음

# 회귀 분석의 사전진단
import pandas as pd
df = pd.read_csv('cars.csv')
df.head() # speed를 독립 변수, dist를 종속변수
import seaborn as sns
sns.regplot('speed', 'dist', lowess=True, data=df) # 산점도 + 추세선

# 극단값 있을 경우 회귀 분석 결과 왜곡될 수 있음. 상자 그림 그려서 극단값 확인
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2) # 1행 2열 형태로 2개 그래프 그림
sns.boxplot('speed', data=df, ax=ax1, orient='v') # ax1 그림. 방향은 수직
ax1.set_title('Speed')
sns.boxplot('dist', data=df, ax=ax2, orient='v') # ax2 그림
ax2.set_title('Distance')

# 선형 회귀 분석은 독립 변수와 종속변수가 정규분포 따를 때 잘 작동. 밀도 플롯 그려서 정규분포의 형태인지 확인. 대체로 중심부에 데이터 몰려있고 좌우로 갈수록 줄어들면 정규분포와 비슷한 형태
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.kdeplot(df['speed'], ax=ax1) # speed의 밀도 플롯
ax1.set_title('Speed')
sns.kdeplot(df['dist'], ax=ax2) # dist의 밀도 플롯
ax1.set_title('Distance')

# 데이터가 치우친 정도를 나타내는 왜도(skewness) 구하기
import scipy.stats
scipy.stats.skew(df['speed'])
scipy.stats.skew(df['dist'])
# +면 오른쪽(높은 값)으로 치우쳤다는 뜻. -면 왼쪽(낮은 값)으로 치우쳤다는 뜻

# 회귀 분석 실시
import pandas as pd
df = pd.read_csv('cars.csv')

# ols 함수로 회귀 분석 실시. 종속변수~독립 변수 형태. y=f(x)처럼
from statsmodels.formula.api import ols
res = ols('dist~speed', data=df).fit()
res.summary() # 결과 확인

# 모형적합도 : 모형이 데이터에 잘 맞는 정도 보여주는 지표들
# R-squared(0.65) : R 제곱. 모형적합도(설명력). dist의 분산을 speed가 약 65% 설명
# F-statistic : 회귀 모형에 대한 통계적 유의미성. 표본뿐만 아니라 모집단에서도 의미 있는 모형
# coef : 데이터로부터 얻은 계수의 측정치
# Intercept는 -17.5791로 speed가 0일 때의 dist의 값
# speed의 계수 추정치는 3.9324로 speed가 1 증가할 때마다 dist가 3.9324 증가
# 즉 dist = -17.5791 + 3.9324 * speed
# F(1,48) = 89.57, p < 0.05는 F(DF Model, DF Residuals) = F-statistic, 그리고 p-값 1.49e-12로 매우 작은 값이므로 p < 0.05라고 할 수 있음
# t(48) = 9.464, p < 0.05에서 t는 알겠고 p는 speed에서 P(>|t|)를 말하는 것임

df.shape # (1599, 13)
df.insert(0, column='type', value='white') # 앞에 값이 모두 white인 컬럼 생김
wine.loc[wine['type'] == 'red', 'quality'] # type이 red인 것 중 quality 컬럼만 뽑아냄

Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid'
model = ols(Rformula, data=wine)
result = model.fit()
result.summary()

# 예측 : 회귀 분석 모델로 품질 등급 예측하기
sample = wine[wine.columns.difference(['quality', 'type'])] # quality, type 제외 나머지
sample = sample[0:5][:] # 0~4행
sample_pred = result.predict(sample)
sample_pred

titanic['age'] = titanic['age'].fillna(titanic['age'].median()) # age 열의 결측값(NaN)을 중앙값으로 채움
sns.countplot('pclass', hue='survived', data=titanic) # pclass는 x축, survived를 pclass 별로 count 해서 그래프 나타냄

# 상관 분석을 위한 상관계수 구하기
titanic.corr(method='pearson') # 어떤 column과 어떤 column과의 상관관계 쭉 나옴
titanic['survived'].corr(titanic['adult_male']) # 특정 변수 사이의 상관계수 구함

# 값을 다른 값으로 매칭
titanic['age2'] = titanic['age'].apply(category_age) # category_age는 함수로 age 값에 따라 if로 다른 값 반환
titanic['sex'] = titanic['sex'].map({'male': 1, 'female': 0})