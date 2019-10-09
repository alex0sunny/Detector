import base64
import json
import zmq
import struct
import numpy as np
import time

from obspy import *
from matplotlib import pyplot

context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.connect('tcp://192.168.0.200:10003')

st = Stream()

pyplot.ion()
figure = pyplot.figure()

cur_time = time.time()
id = socket.recv()
print('id:' + str(id))
if socket.recv() != b'':
    print('no expected emtpy str')
    exit(2)
while True:
    if socket.recv() != id:
        print('incorrect id')
        exit(2)
    else:
        1 == 1
        #print('correct id')
    bdata = socket.recv()
    size_data = bdata[:4]
    size = int.from_bytes(size_data, byteorder='little')
    print('size:' + str(size))
    bdata = bdata[4:]
    bytes_recvd = len(bdata)
    while bytes_recvd < size:
        if socket.recv() != id:
            print('incorrect id')
            exit(2)
        else:
            1 == 1
            #print('correct id')
        bdata += socket.recv()
        bytes_recvd = len(bdata)
    #print('bdata size:' + str(len(bdata)) + '\nbdata:' + str(bdata))
    if bdata[-1] == 125:
        #print('bdata:\n' + str(bdata))
        try:
            json_data = json.loads(bdata.decode('utf-8'))
        except Exception as e:
            print('exception:\n' + str(e))
            print('bdata:\n' + str(bdata))
            exit(2)

        if 'signal' in json_data:
            #print(json_data)
            sampling_rate = json_data['signal']['sample_rate']
            starttime = UTCDateTime(json_data['signal']['timestmp'])
            for ch in json_data['signal']['samples']:
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
            st = st.trim(starttime=st[-1].stats.endtime-5)
            st.merge(fill_value=0)
            if time.time() > cur_time + 1:
                cur_time = time.time()
                pyplot.clf()
                st.plot(fig=figure)
                pyplot.show()
                pyplot.pause(.1)
            #print(st)
    else:
        while bdata[-1] != 125:
            print('skip packet')
            if socket.recv() != id:
                print('incorrect id')
                exit(2)
            else:
                1 == 1
                #print('correct id')
            bdata = socket.recv()

socket.close()