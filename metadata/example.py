import md_intrinio_client

from md_intrinio_client import intrinio_get_company_metadata
from md_intrinio_client import intrinio_get_company_financials
from md_intrinio_client import intrinio_get_company_financials_csv
from md_intrinio_client import get_SandP_metadata
from md_intrinio_client import test_SandP_metadata


#get_SandP_metadata()
test_SandP_metadata()

print("Testing company metadata api")
data = intrinio_get_company_metadata('GE')
print(data)



print("Testing company financials api")
data = intrinio_get_company_financials('GE', '2010', 'FY')
print(data)



print("Testing company financials api via CSV")
data = intrinio_get_company_financials_csv('GE', '2010', 'FY')

print data
my_list = list(data)
print my_list
for row in my_list:
	print(row)
