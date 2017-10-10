from api_key import API_KEY

import json
import urllib.request
from urllib.request import urlopen
req_string= ('https://www.quandl.com/api/v3/datasets/EOD/CSCO.csv?api_key=' + API_KEY).replace(" ", "")
r = urllib.request.urlopen(req_string)
lines = r.readlines()

with open("CSCO_stock_price.txt", "w+") as fw:
	for line in lines:
		fw.write(str(line) + "\n")


# for revenue
req_string= ('https://www.quandl.com/api/v3/datatables/SHARADAR/SF1.csv?ticker=CSCO&qopts.columns=ticker,dimension,datekey,revenue&api_key=' + API_KEY).replace(" ", "")
revenue = urllib.request.urlopen(req_string)

lines = revenue.readlines()

with open("CSCO_revenue.txt", "w+") as fr:
	for line in lines:
		fr.write(str(line) + "\n")
