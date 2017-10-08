import json
import urllib.request
from urllib.request import urlopen
r = urllib.request.urlopen('https://www.quandl.com/api/v3/datasets/EOD/CSCO.csv?api_key=xR-TQ5sb4NnAa2AtX1TG')
lines = r.readlines()

for line in lines:
	print (line)

