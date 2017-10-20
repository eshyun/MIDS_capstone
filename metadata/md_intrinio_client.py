
# coding: utf-8

# In[4]:

from api_key import INTRINIO_USERNAME
from api_key import INTRINIO_PASSWORD                 
import pandas as pd
import sys
import urllib2
import json
import requests


# In[5]:

#print (INTRINIO_USERNAME)
#print(INTRINIO_PASSWORD)


# In[6]:

def intrinio_get_company_metadata(symbol):
        # Get the latest FY Income Statement for "symbol"

        cleanedupdata = {}
        base_url = "https://api.intrinio.com"

        request_url = base_url + "/companies"
        query_params = {
                'ticker': symbol
        }

        response = requests.get(request_url, params=query_params, auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
        if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

        data = response.json()
        return(json.dumps(data, indent=1, sort_keys=True))

#data = intrinio_get_company_metadata('GE')
#print(data)


# In[7]:

def intrinio_get_company_financials(symbol, year, quarter):
        # Get the latest FY Income Statement for "symbol"

        cleanedupdata = {}
        base_url = "https://api.intrinio.com"
        request_url = base_url + "/financials/standardized"
        query_params = {
                'ticker': symbol,
                'statement': 'income_statement',
                'fiscal_year' : str(year),
                'fiscal_period' : str(quarter),
                'type': 'FY'
        }

        response = requests.get(request_url, params=query_params, auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
        if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

        data = response.json()['data']

        for row in data:
                tag = row['tag']
                value = row['value']
                cleanedupdata["AASYMBOL"] = symbol
                cleanedupdata["ABYEAR"] = year
                cleanedupdata["ACPeriod"] = quarter
                cleanedupdata[tag] = value
                #print(tag + ": " + str(value))
        return (json.dumps(cleanedupdata, indent=1, sort_keys=True))

#data = intrinio_get_company_financials('GE', '2010', 'FY')


# In[18]:

intrinio_get_company_financials("GE", "2014", "Q3")


# In[12]:

import csv

def intrinio_get_company_financials_csv(symbol, year, quarter):
        # Get the latest FY Income Statement for "symbol"

        cleanedupdata = {}
        base_url = "https://api.intrinio.com"
        request_url = base_url + "/financials/standardized.csv?"
        query_params = {
                'ticker': symbol,
                'statement': 'income_statement',
                'fiscal_year' : str(year),
                'fiscal_period' : str(quarter),
                'type': 'FY'
        }

        response = requests.get(request_url, params=query_params, auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
        if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

        return(response)


# In[16]:

intrinio_get_company_financials_csv("GE", "2006", "Q3")


# In[5]:

import sys
import json
import finsymbols

sys.path.append('/home/skillachie/Desktop/')
from finsymbols import symbols

sp500 = symbols.get_sp500_symbols()
companies = json.dumps(sp500, indent=1, sort_keys=True)


# In[6]:

# import os
# count = 0
# def get_SandP_metadata():
#     if not os.path.exists("data"):
#         os.makedirs("data")
#     for entry in sp500:
#         datafileName = "./data/data_" + str(entry["symbol"])
#         with open(datafileName, "w") as outfile:
#             data = intrinio_get_company_metadata(entry["symbol"])
#             json.dump(data, outfile, indent=1, sort_keys=True)
#             count += 1


# In[7]:

# # Reading data back
# with open('data/data_MMM', 'r') as f:
#      data = json.load(f)

# print(data)


# In[ ]:

#!jupyter nbconvert --to script md_intrinio_client.ipynb


# In[ ]:

import os
def get_SandP_metadata():
    count = 0
    if not os.path.exists("data"):
        os.makedirs("data")
    for entry in sp500:
        datafileName = "./data/data_" + str(entry["symbol"]+".json")
        with open(datafileName, "w") as outfile:
            data = intrinio_get_company_metadata(entry["symbol"])
            json.dump(data, outfile, indent=1, sort_keys=True)
            count += 1


# Reading data back
def test_SandP_metadata():
        with open('data/data_MMM.json', 'r') as f:
                data = json.load(f)

        print(data)


