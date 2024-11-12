import requests
import pandas as pd
from datetime import datetime, timedelta

api_key = '036ff426e41e82664c78f723b5d08520'

def get_movies_data(api_key, start_date, end_date):
    req_num = 0
    api_url_boxoffice = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
    api_url_movieinfo = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json'

    columns = ['movieCd', 'movieNm', 'openDt', 'rank', 'salesAcc', 'audiAcc', 'typeNm', 'nationNm', 'genreNm', 'watchGradeNm']
    df = pd.DataFrame(columns=columns)
    
    current_date = start_date
    while current_date <= end_date:
        targetDt = current_date.strftime('%Y%m%d')
        url_boxoffice = f'{api_url_boxoffice}?key={api_key}&targetDt={targetDt}'
        
        try:
            response = requests.get(url_boxoffice)
            data = response.json()
            daily_movies = data['boxOfficeResult']['dailyBoxOfficeList']
            req_num += 1
            for dm in daily_movies:
                movieCd = dm['movieCd']
                movieNm = dm['movieNm'] if daily_movies['movieNm'] else None
                openDt = dm['openDt'] if daily_movies['openDt'] else None
                rank = int(dm['rank']) if daily_movies['rank'] else None
                salesAcc = dm['salesAcc'] if daily_movies['salesAcc'] else None
                audiAcc = dm['audiAcc'] if daily_movies['audiAcc'] else None
                
                movie = df.loc[df['movieCd'] == movieCd]
                if not movie.empty:
                    index = movie.index[0]
                    if rank is not None:
                        df.loc[index, 'rank'].append(rank)
                    if salesAcc is not None:
                        df.loc[index, 'salesAcc'] = salesAcc
                    if audiAcc is not None:
                        df.loc[index, 'audiAcc'] = audiAcc
                else:
                    url_movieinfo = f'{api_url_movieinfo}?key={api_key}&movieCd={movieCd}'
                    response = requests.get(url_movieinfo)
                    data = response.json()
                    movie_info = data['movieInfoResult']['movieInfo']
                    req_num += 1
                    typeNm = movie_info['typeNm'] if movie_info['typeNm'] else None
                    nationNm = movie_info['nations'][0]['nationNm'] if movie_info['nations'] else None
                    genres = movie_info['genres']
                    genreNm = [genre['genreNm'] for genre in genres] if genres else None
                    watchGradeNm = movie_info['audits'][0]['watchGradeNm'] if movie_info['audits'] else None
                    
                    movie_df = pd.DataFrame([[movieCd, movieNm, openDt, [rank], salesAcc, audiAcc, typeNm, nationNm, genreNm, watchGradeNm]],
                                            columns=columns);
                    df = pd.concat([df, movie_df], ignore_index=True)
            current_date += timedelta(days=1);
                
        except Exception as e:
            print(f'Error : {e}')
            break
    print(req_num)
    return df
    
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 1)
#end_date = datetime(2023, 12, 31)

movies_df = get_movies_data(api_key, start_date, end_date)

# rank 평균 내서 넣기

movies_df.to_csv('daily.csv', index=False)