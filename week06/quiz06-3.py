from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

#[CODE 1]
def hollys_weather(result):
    Hollys_url = 'https://www.weather.go.kr/w/obs-climate/land/city-obs.do'
    print(Hollys_url)
    html = urllib.request.urlopen(Hollys_url)
    soupHollys = BeautifulSoup(html, 'html.parser')
    tag_tbody = soupHollys.find('tbody')
    for weather in tag_tbody.find_all('tr'):
        if len(weather) <= 3:
            break
        weather
        weather_sido=weather.find('a').string
        weather_td = weather.find_all('td')
        weather_temperature = weather_td[4].string
        weather_humidity = weather_td[9].string
        #result.append([weather_sido]+[weather_temperature]+[weather_humidity])
        result.append([weather_sido, weather_temperature, weather_humidity])
    return

#[CODE 0]
def main():
    result = []
    print('Hollys weather crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    hollys_weather(result)
    hollys_tbl = pd.DataFrame(result, columns=['sido-gu', '온도','습도'])
    del result[:]
    return hollys_tbl
      
if __name__ == '__main__':
    hollys_tbl=main()
    print(hollys_tbl)
    
