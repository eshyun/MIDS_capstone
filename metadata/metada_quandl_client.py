import json
import urllib.request
from urllib.request import urlopen
r = urllib.request.urlopen('https://www.quandl.com/api/v3/datasets/EOD/CSCO.csv?api_key=xR-TQ5sb4NnAa2AtX1TG')
lines = r.readlines()

with open("CSCO_stock_price.txt", "w+") as fw:
	for line in lines:
		fw.write(str(line) + "\n")


# for revenue
revenue = urllib.request.urlopen('https://www.quandl.com/api/v3/datatables/SHARADAR/SF1.csv?ticker=CSCO&qopts.columns=ticker,dimension,datekey,revenue&api_key=xR-TQ5sb4NnAa2AtX1TG')

lines = revenue.readlines()

with open("CSCO_revenue.txt", "w+") as fr:
	for line in lines:
		fr.write(str(line) + "\n")
