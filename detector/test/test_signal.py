import matplotlib.pyplot as plot
from obspy import *
import time

from detector.test.signal_generator import SignalGenerator

plot.ion()
figure = plot.figure()
st = read('D:/converter_data/example/onem.mseed')

signal_generator = SignalGenerator(st)

prev_time = UTCDateTime()
time.sleep(.5)

sig_pool = Stream()

while True:
    sig_pool += signal_generator.get_stream()
    sig_pool = sig_pool.slice(sig_pool[0].stats.endtime - 5).merge()
    print(sig_pool)
    sig_pool.plot(fig=figure)
    plot.show()
    plot.pause(.5)
    plot.clf()

