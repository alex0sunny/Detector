import base64
import json
import socket
import struct
import numpy as np
import time

from obspy import *
from matplotlib import pyplot

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
s.connect(("192.168.0.200", 10003))
#s.connect(("localhost", 5555))

st = Stream()

pyplot.ion()
figure = pyplot.figure()

cur_time = time.time()
while True:
    size_data = s.recv(4)
    size = int.from_bytes(size_data, byteorder='little')
    print('size:' + str(size))
    bdata = b''
    bytes_recvd = 0
    while bytes_recvd < size:
        bdata += s.recv(size - bytes_recvd)
        bytes_recvd = len(bdata)
    #print('bdata size:' + str(len(bdata)) + '\nbdata:' + str(bdata))
    if bdata[-1] == 125:
        json_data = json.loads(bdata.decode('utf-8'))
        if 'signal' in json_data:
            #print(json_data)
            sampling_rate = json_data['signal']['sample_rate']
            starttime = UTCDateTime(json_data['signal']['timestmp'])
            chs = json_data['signal']['samples']
            for ch in chs:
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                #print('bin signal received')
                data = np.frombuffer(bin_signal, dtype='int32')
                #print('data:' + str(data[:100]))
                tr = Trace()
                tr.stats.starttime = starttime
                tr.stats.sampling_rate = sampling_rate
                tr.stats.channel = ch
                tr.data = data
                st += tr
            st.sort()
            st.merge(fill_value='latest')
            st = st.trim(starttime=st[-1].stats.endtime - 5)
            if time.time() > cur_time + 2:
                cur_time = time.time()
                pyplot.clf()
                st.plot(fig=figure)
                pyplot.show()
                pyplot.pause(.01)
                print(st)
        # else:
        #     print('bdata:' + str(bdata))
    else:
        while bdata[-1] != 125:
            print('skip packet')
            bdata = s.recv(10000)

socket.close()