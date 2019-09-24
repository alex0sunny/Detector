import matplotlib.pyplot as plot
from obspy import *
import time

plot.ion()
figure = plot.figure()
st = read('D:/converter_data/example/onem.mseed')
delta = st[0].stats.delta
duration = st[0].stats.endtime - st[0].stats.starttime + st[0].stats.delta
starttime = UTCDateTime(UTCDateTime()._get_ns() // 10**9)
nexttime = starttime + duration
for tr in st:
    tr.stats.starttime = starttime
prev_time = UTCDateTime()
time.sleep(.5)
while True:
    current_time = UTCDateTime()
    if current_time > nexttime - 1:
        starttime = nexttime
        nexttime = starttime + duration
        st_cut = st.slice(starttime - 1)
        for tr in st:
            tr.stats.starttime = starttime
        st = st_cut + st
    st_cur = st.slice(prev_time + delta, current_time)
    print(st_cur)
    st_cur.plot(fig=figure)
    plot.show()
    plot.pause(.5)
    plot.clf()
    prev_time = current_time