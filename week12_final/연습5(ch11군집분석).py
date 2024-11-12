# k-평균 알고리즘 : k개의 중심점을 임의 위치 잡고 중심점 기준으로 가까이 있는 데이터 확인한 뒤 그들과의 거리의 평균 지점으로 중심점 이동 (클러스터의 수=k)
# 엘보 방법 : 왜곡의 변화를 그래프로. 그래프가 꺾이는 지점이 엘보. 그 지점이 최적의 k
# 실루엣 분석 : 클러스터 내에 얼마나 조밀하게 모여있는지 측정. 실루엣 계수 s(i). -1에서 1 값. 1에 가까울수록 좋은 군집화

# 오류 데이터 제거, 정제
data = data[data['Quantity']>0]
data = data[data['CustomerId'].notnull()]
data['CustomerId'] = data['CustomerId'].astype(int)
data.isnull().sum()
data.drop_duplicates(inplace = True) # 중복 레코드 제거

# 분석용 데이터 생성
aggregations = {
	'invoiceNo': 'count',
	'saleAmount': 'sum',
	'invoiceDate': 'max'
}
customer_df = data.groupby('CustomerId').agg(aggregations)
customer_df = customer_df.reset_index()
customer_df = customer_df.rename(columns = {'invoiceNo': 'Freq', 'InvoiceDate': 'ElasedDays'})

import datetime
datetime.datetime(2011, 12, 10)

# k-평균 군집화 모델
distortions = []
for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=0)
    kmeans_i.fit(data_scaled)
    distortions.append(kmeans_i.intertia_)
plt.plot(range(1, 11), distortions, maker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
y_pred = kmeans.fit_predict(data_scaled)
customer_df['ClusterLabel'] = y_pred

# 텍스트 마이닝 : 비정형의 텍스트 데이터로부터 패턴을 찾아내 의미 있는 정보를 추출하는 분석 과정 또는 기법
# 특성 벡터화와 특성 추출 : 텍스트를 구성하는 단어 기반의 특성 추출을 하고 이를 숫자형 값인 벡터값으로 표현해야 함. 특성 벡터화의 대표적인 방법으로 BoW와 Word2ve가 있음
# BOW : 문서가 가지고 있는 모든 단어에 대해 순서는 무시한 채 빈도만 고려하여 단어가 얼마나 자주 등장하는지로 특성 벡터 만드는 방법. 카운트 기반 벡터화와 TF-IDF 기반 벡터화 방식이 있음
# 카운트 기반 벡터화 : 단어에 숫자형 값을 할당할 때 단어 빈도 부여하는 방식. 단어 출현 빈도가 높을수록 중요한 단어로 다루어짐. 사이킷런의 CountVectorizer 모듈에서 제공
# tf(t, d) = 문서 d에 등장한 단어 t의 횟수
# TF-IDF 기반 벡터화 : 특정 문서에 많이 나타나는 단어는 해당 문서의 단어 벡터에 가중치 높임. 모든 문서에 많이 나타나는 단어는 범용적으로 사용하는 단어로 취급하여 가중치 낮추는 방식
# 문서 d에 등장한 단어 t의 TF-IDF = tf(t, d) * idf(t, d)
# 역문서 빈도 : idf(t, d)
# df(d, t) = 단어 t가 포함된 문서 d의 개수