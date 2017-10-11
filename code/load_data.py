import os
import sys
import time
import pandas as pd
import datetime
#import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style
from bs4 import BeautifulSoup
import requests

'''
function to get the stock symbols from S&P 500
'''
def get_SP500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S.26P_500_Component_Stocks"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html5lib")

    table = soup.find_all('table')[0]
    # Exclude header row
    rows = table.find_all('tr')[1:]
    #print(len(rows))
    #print(rows[len(rows)-1])
    stocks = []
    for row in rows:
        cols = row.find_all('td')
        stocks.append(cols[0].get_text())
    return stocks

'''
Get the stock price csv given stock symbol and directory of the csv
'''
def get_csv_file(ticker, base_dir='../data'):
    return os.path.join(base_dir, "{}.csv".format(str(ticker)))

'''
Given a list of stock symbols, start and end date, dump the data in csv format into a directory
'''
def dump_tickers(tickers,
                base_dir='../data',
                data_source='yahoo', # google'
                start_date= '2000-01-01', 
                end_date=datetime.datetime.today().strftime('%Y-%m-%d')):

    for ticker in tickers:
        file = get_csv_file(ticker, base_dir)
        existing_data = None
        # Incremental run: If the csv exists, find the last date and start from there
        try:          
            existing_data = pd.read_csv(file, index_col='Date')

            last_date = existing_data.index[len(existing_data.index)-1]
            print('last_date',last_date)
            new_date = datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)
            new_start_date = (new_date).strftime('%Y-%m-%d')
        except FileNotFoundError as e:
            print(e)
            new_start_date = start_date
        
        #print("existing_data:", existing_data)
        print("new_start_date:", new_start_date)
        attempts = 0
        # Retry 3 times if there is exceptions with reading data
        while attempts < 3:
            try:
                ticker_data = data.DataReader(ticker, data_source, new_start_date, end_date)
                print("Processing", ticker)

                # Concate to existing file
                if (existing_data is not None):
                    with open(file, 'a') as f:
                        ticker_data.to_csv(f, header=False)
                else:   
                    ticker_data.to_csv(file)
                break
            except:
                e = sys.exc_info()[0]
                print("Got", e, ". Retrying....")
                attempts += 1
                