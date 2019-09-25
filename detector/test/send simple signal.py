import socket
import time

from obspy import *

from detector.header_util import chunk_stream, stream_to_bin

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('localhost', 5555))
    s.listen(10)
    conn, addr = s.accept()
    st = read('D:/converter_data/example/onem.mseed')
    with conn:
        print('Connected by', addr)
        while True:
            sts = chunk_stream(st)
            bin_datas = [stream_to_bin(st_) for st_ in sts]
            for bin_data in bin_datas:
                print('bdata size:' + str(len(bin_data)))
                conn.sendall(bin_data)
                time.sleep(.01)
            time.sleep(.1)


