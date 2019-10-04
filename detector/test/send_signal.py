import socket
import time

from obspy import *

from detector.header_util import chunk_stream, stream_to_bin, stream_to_json
from detector.test.signal_generator import SignalGenerator


def send_signal(st, host, port):

    signal_generator = SignalGenerator(st)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(10)
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                st = signal_generator.get_stream()
                sts = chunk_stream(st)
                json_datas = [stream_to_json(st).encode('utf8') for st in sts]
                for json_data in json_datas:
                    data_len = len(json_data)
                    print('bdata size:' + str(data_len))
                    size_bytes = int(data_len).to_bytes(4, byteorder='little')
                    conn.sendall(size_bytes + json_data)
                    time.sleep(.01)
                time.sleep(.1)


st = read('D:/converter_data/example/onem.mseed')
for tr in st:
    tr.stats.station = 'ND01'

send_signal(st, 'localhost', 5555)
