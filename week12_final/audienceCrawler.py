import requests
import pandas as pd
from datetime import datetime, timedelta

api_key = '036ff426e41e82664c78f723b5d08520'
api_url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

df = pd.DataFrame(columns=['movieCd', 'movieNm', 'openDt', 'audiCnt', 'startDt', 'endDt'])
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 2, 1)
#end_date = datetime(2024, 5, 31)
    
while start_date <= end_date:
    targetDt = start_date.strftime('%Y%m%d')
    url = f'{api_url}?key={api_key}&targetDt={targetDt}'
    
    try:
        response = requests.get(url)
        data = response.json()
        daily_movies = data['boxOfficeResult']['dailyBoxOfficeList']
        
        for dm in daily_movies:
            movieCd = dm['movieCd']
            movieNm = dm['movieNm']
            openDt = dm['openDt']
            audiCnt = int(dm['audiCnt'])
            endDt = start_date.strftime('%Y-%m-%d')
            
            movie = df.loc[df['movieCd'] == movieCd]
            if not movie.empty:
                index = movie.index[0]
                df.loc[index, 'audiCnt'].append(audiCnt)
                df.loc[index, 'endDt'] = endDt
            else:
                movie_df = pd.DataFrame([[movieCd, movieNm, openDt, [audiCnt], endDt, endDt]],
                                        columns=['movieCd', 'movieNm', 'openDt', 'audiCnt', 'startDt', 'endDt'])
                df = pd.concat([df, movie_df], ignore_index=True)
            
            start_date += timedelta(days=1)
            
    except Exception as e:
        print(f'Error occurred for date {targetDt}: {e}')
        break
        
df = df.sort_values(by='movieCd')

df.to_csv('audience.csv', index=False)