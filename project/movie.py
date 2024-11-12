import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data
movie = pd.read_csv("movie.csv")

movie.info()
movie['openDate'] = pd.to_datetime(movie['openDate'], errors='coerce')
start_date1 = '2017-01-01'
end_date1 = '2018-12-31'
start_date2 = '2022-06-01'
end_date2 = '2024-05-31'
movie = movie[((movie['openDate'] >= start_date1) & (movie['openDate'] <= end_date1)) |
                       ((movie['openDate'] >= start_date2) & (movie['openDate'] <= end_date2))]
movie = movie[movie['genre'].notna()]
movie = movie[movie['sales'] > 100000]

movie.nation.unique()
movie.nation = movie.nation.replace('한국', 0)
movie.nation = movie.nation.replace('미국', 1)

movie.watchGrade.unique()
movie.watchGrade = movie.watchGrade.replace('전체관람가', 2)
movie.watchGrade = movie.watchGrade.replace('12세이상관람가', 3)
movie.watchGrade = movie.watchGrade.replace('15세이상관람가', 4)
movie.watchGrade = movie.watchGrade.replace('12세이상관람가,15세이상관람가', 4)
movie.watchGrade = movie.watchGrade.replace('청소년관람불가', 5)

movie.genre.unique()
movie.genre = movie.genre.apply(lambda g : g.split(',')[0])

movie = movie.sort_values(by='sales', ascending=False).reset_index(drop=True)


# chart
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10, 6))
plt.plot(movie.sales.values)
plt.ticklabel_format(style='plain', axis='y')
plt.title('전체 매출액')
plt.xlabel('영화')
plt.ylabel('매출액')
plt.show()

genre_sales_sum = movie.groupby('genre')['sales'].sum().reset_index()
genre_sales_sum = genre_sales_sum.sort_values(by='sales', ascending=False).reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.bar(genre_sales_sum['genre'], genre_sales_sum['sales'])
plt.xlabel('장르')
plt.ylabel('전체 매출액')
plt.title('장르 별 전체 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

genre_sales_mean = movie.groupby('genre')['sales'].mean().reset_index()
genre_sales_mean = genre_sales_mean.sort_values(by='sales', ascending=False).reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.bar(genre_sales_mean['genre'], genre_sales_mean['sales'])
plt.xlabel('장르')
plt.ylabel('평균 매출액')
plt.title('장르 별 평균 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

genre_dummies = pd.get_dummies(movie['genre'])
genre_sales = pd.concat([genre_dummies, movie['sales']], axis=1)
correlation = genre_sales.corr()
genre_sales_corr = correlation[['sales']].drop('sales')
plt.figure(figsize=(10, 8))
sns.heatmap(genre_sales_corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, annot_kws={"size":10})
plt.title('장르 별 매출액과 상관관계')
plt.show()

nation_sales_sum = movie.groupby('nation')['sales'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(nation_sales_sum['nation'], nation_sales_sum['sales'])
plt.xlabel('국가')
plt.ylabel('전체 매출액')
plt.title('국가 별 전체 매출액(한국:0, 미국:1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

nation_sales_mean = movie.groupby('nation')['sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(nation_sales_mean['nation'], nation_sales_mean['sales'])
plt.xlabel('국가')
plt.ylabel('평균 매출액')
plt.title('국가 별 평균 매출액(한국:0, 미국:1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

watch_grade_sales_sum = movie.groupby('watchGrade')['sales'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(watch_grade_sales_sum['watchGrade'], watch_grade_sales_sum['sales'])
plt.xlabel('관람등급')
plt.ylabel('전체 매출액')
plt.title('관람등급 별 전체 매출액(2:전체관람가, 3:12세관람가, 4:15세관람가, 5:청소년관람불가)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

watch_grade_sales_mean = movie.groupby('watchGrade')['sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(watch_grade_sales_mean['watchGrade'], watch_grade_sales_mean['sales'])
plt.xlabel('관람등급')
plt.ylabel('평균 매출액')
plt.title('관람등급 별 평균 매출액(2:전체관람가, 3:12세관람가, 4:15세관람가, 5:청소년관람불가)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

watch_grade_dummies = pd.get_dummies(movie['watchGrade'])
watch_grade_sales = pd.concat([watch_grade_dummies, movie['sales']], axis=1)
correlation = watch_grade_sales.corr()
watch_grade_sales_corr = correlation[['sales']].drop('sales')
plt.figure(figsize=(10, 8))
sns.heatmap(watch_grade_sales_corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, annot_kws={"size":30}, square=True)
plt.title('관람등급 별 매출액과 상관관계')
plt.show()

screens_sales_sum = movie.groupby('screens')['sales'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(screens_sales_sum['screens'], screens_sales_sum['sales'])
plt.xlabel('스크린수')
plt.ylabel('전체 매출액')
plt.title('스크린수 별 전체 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

screens_sales_mean = movie.groupby('screens')['sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(screens_sales_mean['screens'], screens_sales_mean['sales'])
plt.xlabel('스크린수')
plt.ylabel('평균 매출액')
plt.title('스크린수 별 평균 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def get_month(month):
    if month in [12, 1, 2]:
        return '12월~2월'
    elif month in [3, 4, 5]:
        return '3월~5월'
    elif month in [6, 7, 8]:
        return '6월~8월'
    elif month in [9, 10, 11]:
        return '9월~11월'

movie['month'] = movie['openDate'].dt.month
movie['month'] = movie['month'].apply(get_month)
open_date_sales_sum = movie.groupby('month')['sales'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(open_date_sales_sum['month'], open_date_sales_sum['sales'])
plt.xlabel('분기')
plt.ylabel('전체 매출액')
plt.title('분기 별 전체 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

open_date_sales_mean = movie.groupby('month')['sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(open_date_sales_mean['month'], open_date_sales_mean['sales'])
plt.xlabel('분기')
plt.ylabel('평균 매출액')
plt.title('분기 별 평균 매출액')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

month_dummies = pd.get_dummies(movie['month'])
month_sales = pd.concat([month_dummies, movie['sales']], axis=1)
correlation = month_sales.corr()
month_sales_corr = correlation[['sales']].drop('sales')
plt.figure(figsize=(10, 8))
sns.heatmap(month_sales_corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, annot_kws={"size":10})
plt.title('분기 별 매출액과 상관관계')
plt.show()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
movie['genre'] = encoder.fit_transform(movie['genre'])
movie['month'] = encoder.fit_transform(movie['month'])
        
heatmap_movie = movie[['sales', 'genre', 'nation', 'screens', 'watchGrade', 'month']]
sns.heatmap(heatmap_movie.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True,
            cmap="YlGnBu", annot=True, annot_kws={"size":10})
plt.show()


# model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

percent_10 = movie['sales'].quantile(0.90)
movie['target'] = (movie['sales'] >= percent_10).astype(int)

scaler = StandardScaler()
movie[['screens']] = scaler.fit_transform(movie[['screens']])

m_train, m_test = train_test_split(movie, test_size=0.2, random_state=10)

model = LogisticRegression()
model.fit(X=pd.DataFrame(m_train[['screens', 'genre', 'nation', 'watchGrade', 'month']]), y=m_train['target'])

m_predict = model.predict(pd.DataFrame(m_test[['screens', 'genre', 'nation', 'watchGrade', 'month']]))

accuracy = accuracy_score(m_test['target'], m_predict) 
print(accuracy)
