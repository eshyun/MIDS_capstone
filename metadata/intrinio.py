from api_key import INTRINIO_USERNAME
from api_key import INTRINIO_PASSWORD
#import intriniorealtime
import intrinio
import pandas as pd
import sys
import urllib2
import json
import requests

# Setup common variables

api_username = INTRINIO_USERNAME
api_password = INTRINIO_PASSWORD
base_url = "https://api.intrinio.com"

# Get the latest FY Income Statement for AAPL

ticker = "AAPL"
request_url = base_url + "/financials/standardized"
query_params = {
    'ticker': ticker,
    'statement': 'income_statement',
    'fiscal_year' : '2010',
    'fiscal_period' : 'Q1',
    'type': 'FY'
}

response = requests.get(request_url, params=query_params, auth=(api_username, api_password))
if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

data = response.json()['data']

for row in data:
    tag = row['tag']
    value = row['value']
    print(tag + ": " + str(value))
