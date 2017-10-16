import sys
import json
import finsymbols

sys.path.append('/home/skillachie/Desktop/')
from finsymbols import symbols

SandP500 = {}

with open("SandP500_symbols.txt", "r") as fr:
	for line in fr:
		company = json.loads(line)
		SandP500[company["symbol"]] = line

for item, value  in SandP500.items():
	print (item, value)
