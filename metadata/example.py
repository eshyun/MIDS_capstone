
# coding: utf-8

# # For Generation of Intrinio based Financial Data - Shankar

# In[1]:

get_ipython().system('pwd')


# In[2]:

from api_key import INTRINIO_USERNAME
from api_key import INTRINIO_PASSWORD             

import md_intrinio_client
import pandas as pd
import json

from md_intrinio_client import intrinio_get_company_metadata
from md_intrinio_client import intrinio_get_company_financials
from md_intrinio_client import intrinio_get_company_financials_csv
from md_intrinio_client import get_SandP_metadata
from md_intrinio_client import test_SandP_metadata
from finsymbols import symbols

import sys
import json
import finsymbols
import ast
import requests


# In[15]:

import json 
SandP500 = {}
companyList = []
with open("SandP500_symbols.txt", "r") as fr:
            for line in fr:
                    company = json.loads(line)
                    SandP500[company["symbol"]] = line
                    companyList.append(company["symbol"])

tickerchunks = [companyList[x:x+95] for x in xrange(0, len(companyList), 95)]


# In[16]:

print(len(tickerchunks))


# In[26]:

print(len(tickerchunks[1]))
print((tickerchunks[1]))


# In[ ]:

# Stopped at EOG


# In[6]:

fy_url1 = "http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t="
fy_url2= "&reportType=is&period=12&dataType=A&order=asc&columnYear=5&number=3"
months =["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


# In[7]:

import re
import urllib
import time
def get_fiscal_year_end():
    loopCount = 0
    with open("fy.csv", "w") as fw:
        for company in companyList:
            print("working on {}".format(company))
            fy_url = fy_url1+company+fy_url2
            f = urllib.urlopen(fy_url)
            time.sleep(1)
            myfile = f.read()
            for line in myfile:
                month = re.search("Fiscal year ends in (.*)", myfile)
                month = str(month.groups()).split(".")[0]
                month = re.sub(r"^\W+", "", month)
                month = month.lower()
            print("writing")
            fw.write("{},FY,{}\n".format(company, month))
            fw.write("{},Q1,{}\n".format(company, months[months.index(month) - 9]))
            fw.write("{},Q2,{}\n".format(company, months[months.index(month) - 6]))
            fw.write("{},Q3,{}\n".format(company, months[months.index(month) - 3]))
            fw.write("{},Q4,{}\n".format(company, month))
            loopCount += 1

            if loopCount > 529:
                return


# In[8]:

#get_fiscal_year_end()


# In[9]:

def new_intrinio_get_company_financials(symbol, year, quarter):
        # Get the latest FY Income Statement for "symbol"
        # 'type': 'FY'
        cleanedupdata = {}
        base_url = "https://api.intrinio.com"
        request_url = base_url + "/financials/standardized"
        query_params = {
                'ticker': symbol,
                'statement': 'income_statement',
                'fiscal_year' : str(year),
                'fiscal_period' : quarter
        }

        response = requests.get(request_url, params=query_params, auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
        if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

        if response.status_code == 429:
            print("API query limit reached")
            return
        data = response.json()['data']


        for row in data:
                tag = row['tag']
                value = row['value']
                cleanedupdata["AASYMBOL"] = symbol
                cleanedupdata["ABYEAR"] = year
                cleanedupdata["ACPeriod"] = quarter
                cleanedupdata[tag] = value

        datalist=[]
        for key, value in sorted(cleanedupdata.items()):
            datalist.append(str(value))
        

        return(datalist)


# In[10]:

def updated_cleanupdata(cleanedupdata):
    newData = {}
    newData['AASYMBOL'] = cleanedupdata['AASYMBOL']
    newData['ABYEAR'] = cleanedupdata['ABYEAR']
    newData['ACPeriod'] = cleanedupdata['ACPeriod']
    newData['basicdilutedeps'] = cleanedupdata.get('basicdilutedeps', 0.0)
    newData['basiceps'] = cleanedupdata.get('basiceps', 0.0)
    #print("basiceps is {}".format(newData['basiceps']))
    
    newData['cashdividendspershare'] = cleanedupdata.get('cashdividendspershare', 0.0)
    newData['dilutedeps'] = cleanedupdata.get('dilutedeps', 0.0)
    newData['incometaxexpense'] = cleanedupdata.get('incometaxexpense', 0.0)
    newData['netincome'] = cleanedupdata.get('netincome', 0.0)
    #print("netincome is {}".format(newData['netincome']))
    newData['netincomecontinuing'] = cleanedupdata.get('netincomecontinuing', 0.0)
    
    newData['netincomediscontinued'] = cleanedupdata.get('netincomediscontinued', 0.0)
    newData['netincometocommon'] = cleanedupdata.get('netincometocommon', 0.0)
    newData['netincometononcontrollinginterest'] = cleanedupdata.get('netincometononcontrollinginterest', 0.0)
    newData['operatingcostofrevenue'] = cleanedupdata.get('operatingcostofrevenue', 0.0)
    newData['operatingrevenue'] = cleanedupdata.get('operatingrevenue', 0.0)
    #print("operatingrevenue is {}".format(newData['operatingrevenue']))
    
    newData['othercostofrevenue'] = cleanedupdata.get('othercostofrevenue', 0.0)
    newData['otherincome'] = cleanedupdata.get('otherincome', 0.0)
    newData['preferreddividends'] = cleanedupdata.get('preferreddividends', 0.0)
    newData['sgaexpense'] = cleanedupdata.get('sgaexpense', 0.0)
    newData['totalcostofrevenue'] = cleanedupdata.get('totalcostofrevenue', 0.0)
    
    newData['totalgrossprofit'] = cleanedupdata.get('totalgrossprofit', 0.0)
    #print("totalgrossprofit is {}".format(newData['totalgrossprofit']))
    newData['totalinterestexpense'] = cleanedupdata.get('totalinterestexpense', 0.0)
    newData['totaloperatingexpenses'] = cleanedupdata.get('totaloperatingexpenses', 0.0)
    newData['totaloperatingincome'] = cleanedupdata.get('totaloperatingincome', 0.0)
    newData['totalotherincome'] = cleanedupdata.get('totalotherincome', 0.0)
    
    newData['totalpretaxincome'] = cleanedupdata.get('totalpretaxincome', 0.0)
    newData['totalrevenue'] = cleanedupdata.get('totalrevenue', 0.0)
    #print("totalrevenue is {}".format(newData['totalrevenue']))
    newData['weightedavebasicdilutedsharesos'] = cleanedupdata.get('weightedavebasicdilutedsharesos', 0.0)
    newData['weightedavebasicsharesos'] = cleanedupdata.get('weightedavebasicsharesos', 0.0)
    newData['weightedavedilutedsharesos'] = cleanedupdata.get('weightedavedilutedsharesos', 0.0)
    
    #newData[] = cleanedupdata.get(, 0.0)
    return newData


# In[11]:

attributes = ["ticker", "year", "quarter", "basicdilutedeps", "basiceps",
              'cashdividendspershare', 'dilutedeps', 'incometaxexpense', 'netincome', 'netincomecontinuing',
              'netincomediscontinued', 'netincometocommon', 'netincometononcontrollinginterest',  
                  'operatingcostofrevenue', 'operatingrevenue',
              'othercostofrevenue', 'otherincome', 'preferreddividends', 'sgaexpense', 'totalcostofrevenue',
              'totalgrossprofit', 'totalinterestexpense', 'totaloperatingexpenses', 'totaloperatingincome', 'totalotherincome', 
              'totalpretaxincome', 'totalrevenue', 'weightedavebasicdilutedsharesos', 'weightedavebasicsharesos', 'weightedavedilutedsharesos'
                 ]
print(len(attributes))

xx = ",".join(attributes)
print(xx)


# In[17]:

def updated_intrinio_get_company_financials(symbol, year, quarter):
        # Get the latest FY Income Statement for "symbol"
        # 'type': 'FY'
        cleanedupdata = {}
        base_url = "https://api.intrinio.com"
        request_url = base_url + "/financials/standardized"
        query_params = {
                'ticker': symbol,
                'statement': 'income_statement',
                'fiscal_year' : str(year),
                'fiscal_period' : quarter
        }

        response = requests.get(request_url, params=query_params, auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
        if response.status_code == 401: print("Unauthorized! Check your username and password."); exit()

        if response.status_code == 429:
            print("API query limit reached")
            return
        data = response.json()['data']

        #print(data['basicdilutedeps'])
        for row in data:
                #print(row)
                tag = row['tag']
                value = row['value']

                cleanedupdata[tag] = value

        datalist=[]
        attr = []
        cleanedupdata["AASYMBOL"] = symbol
        cleanedupdata["ABYEAR"] = year
        cleanedupdata["ACPeriod"] = quarter
        cleanedupdata = updated_cleanupdata(cleanedupdata)
        for key, value in sorted(cleanedupdata.items()):
            datalist.append(str(value))
            attr.append(str(key))
 
        return(datalist, attr)
data = updated_intrinio_get_company_financials('GE', '2008', 'Q1')


# In[18]:

get_ipython().system('pwd')
get_ipython().magic('cd ../data/nlp_by_company')
import glob
xx = list(glob.glob("*.csv"))
print(xx)
yy = []

for item in xx:
    yy.append(str(item).strip('.csv'))

print(yy)
get_ipython().magic('cd ../../metadata')


# In[19]:

company = ['COL', 'CRM', 'DGX', 'FOX', 'FOXA', 'FTI', 'JWN', 'KORS', 'LUV', 'M', 'MA', 'MAA', 
               'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 
               'MHK', 'MKC', 'MLM', 'MMC', 'MNST', 'MON', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSFT', 
               'MSI', 'MTD', 'MU', 'MYL', 'NAVI', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 
               'NLSN', 'NOC', 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWS', 'NWSA', 'O', 
               'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PCAR', 'PCG', 'PCLN', 'PDCO', 
               'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 
               'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD', 
               'PYPL', 'Q', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RHT', 'RJF', 'RL', 
               'RMD', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCG', 'SEE', 'SHW', 
               'SIG', 'SLB', 'SLG', 'SNA', 'SNI', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRCL', 'SRE', 'STI', 
               'STT', 'STX', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYMC', 'SYY', 'TAP', 'TDG', 'TEL', 'TGT', 
               'TIF', 'TJX', 'TMK', 'TMO', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWX', 'TXN', 
               'TXT', 'UDR', 'ULTA', 'USB']

print(len(company))


# In[27]:

def generate_financial_data():
    SandP500 = {}
    companyList = []
    with open("SandP500_symbols.txt", "r") as fr:
            for line in fr:
                    company = json.loads(line)
                    SandP500[company["symbol"]] = line
                    companyList.append(company["symbol"])

    tickerchunks = [companyList[x:x+95] for x in xrange(0, len(companyList), 95)]


    nlp_companies = ['COL', 'CRM', 'DGX', 'FOX', 'FOXA', 'FTI', 'JWN', 'KORS', 'LUV', 'M', 'MA', 'MAA', 
               'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 
               'MHK', 'MKC', 'MLM', 'MMC', 'MNST', 'MON', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSFT', 
               'MSI', 'MTD', 'MU', 'MYL', 'NAVI', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 
               'NLSN', 'NOC', 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWS', 'NWSA', 'O', 
               'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PCAR', 'PCG', 'PCLN', 'PDCO', 
               'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 
               'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD', 
               'PYPL', 'Q', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RHT', 'RJF', 'RL', 
               'RMD', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCG', 'SEE', 'SHW', 
               'SIG', 'SLB', 'SLG', 'SNA', 'SNI', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRCL', 'SRE', 'STI', 
               'STT', 'STX', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYMC', 'SYY', 'TAP', 'TDG', 'TEL', 'TGT', 
               'TIF', 'TJX', 'TMK', 'TMO', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWX', 'TXN', 
               'TXT', 'UDR', 'ULTA', 'USB']
    years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    quarters = ["Q1", "Q2", "Q3", "Q4", "FY"]
    #years = [2010]
    #quarters = ["Q1", "Q2"]
    #companies = ["GE", "CSCO", "GOOG", "FACE"]

    bigDict = {}

    attributes = ["ticker", "year", "quarter", "basicdilutedeps", "basiceps",
              'cashdividendspershare', 'dilutedeps', 'incometaxexpense', 'netincome', 'netincomecontinuing',
              'netincomediscontinued', 'netincometocommon', 'netincometononcontrollinginterest',  
                  'operatingcostofrevenue', 'operatingrevenue',
              'othercostofrevenue', 'otherincome', 'preferreddividends', 'sgaexpense', 'totalcostofrevenue',
              'totalgrossprofit', 'totalinterestexpense', 'totaloperatingexpenses', 'totaloperatingincome', 'totalotherincome', 
              'totalpretaxincome', 'totalrevenue', 'weightedavebasicdilutedsharesos', 'weightedavebasicsharesos', 'weightedavedilutedsharesos'
                 ]
    
    #specific_company = ['TXN', 'TXT', 'UDR', 'ULTA', 'USB']
    print(len(attributes))

    xx = ",".join(attributes) + "\n"

    # tickerchunks[1], tickerchunks[2], tickerchunks[3] and tickerchunks[5] are done
    # Re-doing 0 block, 3 is complete, 2 is complete, do 1 and 5 today. 
    loopIndex = 0
    for company in tickerchunks[1]:
    #for company in specific_company:
    #for company in nlp_companies:
        #with open("../data/nlp_by_company/revenue/"+company+"_Financials_by_Quarter.csv", 'w') as fw:
        with open("./revenue/"+company+"_Financials_by_Quarter.csv", 'w') as fw:
            fw.write(xx)
            print("working on {} - {}".format(company, loopIndex))
#             if loopIndex > 2:
#                 break
            loopIndex += 1
            #print("working on company {}".format(company))
            for year in years:

                #print("working on year {}".format(year))
                for quarter in quarters:
                    #print("working on quarter {}".format(quarter))

                    data, _ = updated_intrinio_get_company_financials(company, str(year), quarter)
                    
                    # Convert list to string
                    datastring = ",".join(data)
                    
                    # Add a linefeed so that every data point is on a different line
                    datastring +="\n"

                    # Write the data to the open file for this company
                    fw.write(datastring)
                    #break
                #break
            #break
        #break

            print("DONE with {}".format(company))
            #return


# In[28]:

get_ipython().system('pwd')


# In[29]:

generate_financial_data()


# In[56]:

import os
os.chdir("/Users/NatarajanShankar/UC_Berkeley/Final_term/Capstone/W210/MIDS_capstone/metadata")
get_ipython().system('pwd')


# In[52]:

import glob, os
import ast

def convert_json_to_csv():
    companyDict = {}
    os.chdir("./data")
    #print("got here")
    for datafile in glob.glob("*Financials*.json"):
        #print(datafile)
        with open(datafile, "r") as fr:
            #print ("Processing {}".format(datafile))
            for line in fr:
                companyDict = ast.literal_eval(line)
                
                # Save company ticker
                for company, companyvalue in companyDict.items():
                    #print("Company is")
                    #print("{}".format(company, end=''))
                    # For every year subset
                    if isinstance(companyvalue, dict):
                        for year, yearvalue in companyvalue.items():
                            #print("Year is")
                            #print("{}".format(year, end=''))
                            # Look at the corresponding quarter
                            if isinstance(yearvalue, dict):
                                datalist=[]
                                # Look at company quarter
                                for quarter, quartervalue in yearvalue.items():
                                    #print("Quarter is")
                                    #print("{}".format(quarter, end=''))
                                    for financialdata in quartervalue.values():
                                        datalist.append(financialdata)
                                    
                                    if len(datalist) > 21:
                                        datalist[14] = "IGNORE"
                                        datalist[21] = "IGNORE"
                                    
                                    if len(datalist) == 2:
                                        datalist[0] = ""
                                        datalist[1] = ""
                                    print("{},{},{},{}".format(company, year, quarter, ",".join(datalist), end=''))
                                    
                                    datalist = []
        break
    
    


# In[53]:

convert_json_to_csv()


# In[ ]:

def get_company_financial_data_all(symbol):
    companyDict = {}
    with open("./data/Financials.json", "r") as fp:
        #companyDict = json.load(fp)
        for line in fp:
            companyDict = ast.literal_eval(line)

        for ticker in companyDict.keys():
            if ticker == symbol:
                print symbol, companyDict[symbol]
            


# In[ ]:

get_company_financial_data_all("GE")


# In[ ]:

def get_company_financial_data_by_year(symbol, year):
    companyDict = {}
    with open("./data/Financials.json", "r") as fp:
        #companyDict = json.load(fp)
        for line in fp:
            companyDict = ast.literal_eval(line)

        for ticker in companyDict.keys():
            if ticker == symbol:
                print symbol, companyDict[symbol][year]
            


# In[ ]:

get_company_financial_data_by_year("GE", "2014")


# In[ ]:

def get_company_financial_data_by_year_and_attr(symbol, year, attribute):
    companyDict = {}
    with open("./data/Financials.json", "r") as fp:
        #companyDict = json.load(fp)
        for line in fp:
            companyDict = ast.literal_eval(line)

        for ticker in companyDict.keys():
            if ticker == symbol:
                print symbol, companyDict[symbol][year][attribute]
            


# In[ ]:

get_company_financial_data_by_year_and_attr("GE", "2014", "basiceps")


# In[ ]:

def generate_company_financial_data_csv(symbol):
    companyDict = {}
    with open("./data/Financials.json", "r") as fp:
        for line in fp:
            companyDict = ast.literal_eval(line)
            
        for ticker in companyDict.keys():
            if ticker == symbol:
                print symbol, companyDict
            
            
            

