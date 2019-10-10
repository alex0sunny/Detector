import base64
import json
import zmq
import struct
import numpy as np
import time

from obspy import *
from matplotlib import pyplot

from detector.StaLtaTrigger import logger


class ZmqReceiver:

    def __init__(self, conn_str):
        self.id = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.connect(conn_str)
        id_bytes = self.socket.recv()
        logger.debug('id bytes: ' + str(id_bytes))
        if len(id_bytes) != 5:
            raise Exception('unexpected id_bytes len:' + str(len(id_bytes)))
        self.id = int.from_bytes(id_bytes, byteorder='big')
        logger.info('id:' + str(self.id))
        empty_data = self.socket.recv()
        if empty_data:
            raise Exception('empty data expected:' + str(empty_data))

    def recv(self):
        id_bytes = self.socket.recv()
        if len(id_bytes) != 5:
            logger.error('unexpected id_bytes len:' + str(len(id_bytes)))
            return id_bytes
        id_ = int.from_bytes(id_bytes, byteorder='big')
        if self.id != id_:
            logger.warning('unexpected id:' + str(id_))
        data = self.socket.recv()
        if not data:
            logger.error('empty data received')
        return data

    def __del__(self):
        self.socket.close()
        self.context.destroy()


class ReceiverBuffer:

    def __init__(self, conn_str):
        self.receiver = ZmqReceiver(conn_str)
        self.buf = b''

    def expand(self, n):
        while len(self.buf) < n:
            self.buf += self.receiver.recv()
        return self.buf

    def shift(self, n=1):
        if self.buf:
            #logger.info('drop bytes, number of bytes:' + str(n))
            self.buf = self.buf[n:]


class NdasBuffer:

    def __init__(self, conn_str):
        self.buffer = ReceiverBuffer(conn_str)

    def recv(self):
        while True:
            self.buffer.expand(5)
            if self.buffer.buf[4:5] != b'{':
                logger.warning('no json start symbol:' + str(self.buffer.buf[:5]))
                self.buffer.shift()
                continue
            size_bytes = self.buffer.buf[:4]
            size = int.from_bytes(size_bytes, byteorder='little')
            if not 20 < size < 20000:
                logger.error('possibly incorrect size:' + str(size))
                self.buffer.shift()
                continue
            self.buffer.expand(size + 4)
            last_byte_bstr = self.buffer.buf[size + 3:size + 4]
            if last_byte_bstr != b'}':
                logger.error('incorrect last byte:' + str(last_byte_bstr))
                self.buffer.shift()
                continue
            bdata = self.buffer.buf[4:size + 4]
            try:
                json_data = json.loads(bdata.decode('utf-8'))
            except Exception as e:
                logger.error('cannot parse json data:\n' + str(bdata) + '\n' + str(e))
                self.buffer.shift()
                continue
            self.buffer.shift(size + 4)
            return json_data

pyplot.ion()
figure = pyplot.figure()

cur_time = time.time()
st = Stream()

receiver = NdasBuffer('tcp://192.168.0.200:10003')
while True:
    json_data = receiver.recv()
    if 'signal' in json_data:
        # print(json_data)
        sampling_rate = json_data['signal']['sample_rate']
        starttime = UTCDateTime(json_data['signal']['timestmp'])
        for ch in json_data['signal']['samples']:
            bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
            # print('bin signal received')
            data = np.frombuffer(bin_signal, dtype='int32')
            # print('data:' + str(data[:100]))
            tr = Trace()
            tr.stats.starttime = starttime
            tr.stats.sampling_rate = sampling_rate
            tr.stats.channel = ch
            tr.data = data
            st += tr
        if time.time() > cur_time + 2:
            st.sort()
            st.merge(fill_value='latest')
            st = st.trim(starttime=st[-1].stats.endtime - 5)
            cur_time = time.time()
            pyplot.clf()
            st.plot(fig=figure)
            pyplot.show()
            pyplot.pause(.1)

