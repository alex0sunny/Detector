from functools import reduce

from obspy import *

st = read()
sts = [Stream(trs) for trs in zip(*[tr / 3 for tr in st])]
[print(st) for st in sts]

