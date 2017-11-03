import md_intrinio_client
import pandas as pd
import json

from md_intrinio_client import intrinio_get_company_metadata
from md_intrinio_client import intrinio_get_company_financials
from md_intrinio_client import intrinio_get_company_financials_csv
from md_intrinio_client import get_SandP_metadata
from md_intrinio_client import test_SandP_metadata


#get_SandP_metadata()
#test_SandP_metadata()

#print("Testing company metadata api")
##data = intrinio_get_company_metadata('GE')
#print(data)



#print("Testing company financials api")
#data = intrinio_get_company_financials('GE', '2010', 'FY')
#print(data)




import sys
import json
import finsymbols

sys.path.append('/home/skillachie/Desktop/')
from finsymbols import symbols

SandP500 = {}
companyList = []
with open("SandP500_symbols.txt", "r") as fr:
        for line in fr:
                company = json.loads(line)
                SandP500[company["symbol"]] = line
		companyList.append(company["symbol"])


#print("Testing company financials api via CSV")

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
#companies = ["GE", "CSCO", "GOOG", "FACE"]

bigDict = {}

for company in companyList:
	bigDict[company] = {}
	for year in years:
		bigDict[company][year] = {}
		data = intrinio_get_company_financials_csv(company, str(year), 'FY')

		financials = list(data)

		#print (financials)
		breakdown = ''.join(financials)
		#print(breakdown)
		newdata = breakdown.split("\n")

		for item in newdata:
			splititem = item.split(",")
			if len(splititem) < 2:
				continue
			bigDict[company][year][splititem[0]]=  {}
			bigDict[company][year][splititem[0]]= splititem[1]

with open("./data/"+"Financials.json", 'w') as file:
	file.write(json.dumps(bigDict))


#print bigDict
