import md_intrinio_client

from md_intrinio_client import intrinio_get_company_metadata
from md_intrinio_client import intrinio_get_company_financials


print("Testing company metadata api")
data = intrinio_get_company_metadata('GE')
print(data)



print("Testing company financials api")
data = intrinio_get_company_financials('GE', '2010', 'FY')
print(data)
