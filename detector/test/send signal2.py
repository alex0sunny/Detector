import socket
import time

from obspy import *

from detector.misc.header_util import chunk_stream, stream_to_bin
from detector.test.signal_generator import SignalGenerator

st = read('D:/converter_data/example/threeCh100.mseed')
st = Stream() + st[0] + st[-1]
for tr in st:
    tr.stats.station = 'ND02'
signal_generator = SignalGenerator(st)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('localhost', 5565))
    s.listen(10)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            st = signal_generator.get_stream()
            sts = chunk_stream(st)
            bin_datas = [stream_to_bin(st) for st in sts]
            for bin_data in bin_datas:
                print('bdata size:' + str(len(bin_data)))
                conn.sendall(bin_data)
                time.sleep(.02)
            time.sleep(.5)


