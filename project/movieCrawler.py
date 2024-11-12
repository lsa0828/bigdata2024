from bs4 import BeautifulSoup
import pandas as pd

def get_movies(file):
    soup = BeautifulSoup(file, 'html.parser')

    movies = []

    table = soup.find('table', class_='tbl_exc')

    for row in table.find_all('tr')[1:-1]:
        columns = row.find_all('td')
        
        movie_name = columns[1].text.strip()
        open_date = columns[2].text.strip()
        sales = columns[3].text.strip()
        screens = columns[8].text.strip()
        nation = columns[10].text.strip()
        genre = columns[15].text.strip()
        watch_grade = columns[14].text.strip()
        
        if nation in ['한국', '미국']:
            movies.append({
                'movieName': movie_name,
                'openDate': open_date,
                'sales': sales,
                'screens': screens,
                'nation': nation,
                'genre': genre,
                'watchGrade': watch_grade
            })
        
    return movies

file1 = open('KOBIS_boxoffice.html', encoding='utf-8').read()
file2 = open('KOBIS_boxoffice2.html', encoding='utf-8').read()

movies1 = get_movies(file1)
movies2 = get_movies(file2)

movies = movies1 + movies2
        
df = pd.DataFrame(movies)
df.to_csv('movie.csv', index=False, encoding='utf-8')
