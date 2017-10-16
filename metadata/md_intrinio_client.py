from api_key import INTRINIO_USERNAME
from api_key import INTRINIO_PASSWORD
#import intriniorealtime
import intrinio
import pandas as pd
import sys
import urllib2
import json
#import urllib.request
#from urllib.request import urlopen

print (INTRINIO_USERNAME)
print(INTRINIO_PASSWORD)

intrinio.client.username = INTRINIO_USERNAME
intrinio.client.password = INTRINIO_PASSWORD


def intrinio_get_stock_price(symbol, start_date, end_date):
	intrinio.client.username = INTRINIO_USERNAME
	intrinio.client.password = INTRINIO_PASSWORD
	xx = intrinio.prices('AAPL', start_date, end_date)

	return xx


def intrinio_get_company_info(symbol):
	intrinio.client.username = INTRINIO_USERNAME
	intrinio.client.password = INTRINIO_PASSWORD
	#https://api.intrinio.com/companies?identifier={symbol}
	#req_string= ('https://api.intrinio.com/companies?identifier=' + symbol).replace(" ", "")

	#print (req_string)

	#r = urllib2.urlopen(req_string)
	#r = urllib2.urlopen(req_string).read()
	#lines = r.readlines()
	lines = intrinio.companies(symbol)

	xx = lines.to_dict()

	#xx = json.load(lines)
	#lines = r.read()

	print((xx["employees"])[0])

	#for line, value in lines:	
		#print (line, value)

	#r.close()
	

intrinio_get_company_info('CSCO')

	
