import sys
import json
import finsymbols

sys.path.append('/home/skillachie/Desktop/')
from finsymbols import symbols

sp500 = symbols.get_sp500_symbols()

with open("SandP500_symbols.txt", "w") as fw:
	for item in sp500:
		json.dump(item, fw)
		fw.write("\n")
