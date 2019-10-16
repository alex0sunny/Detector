import base64
import json

import zmq
from matplotlib import pyplot
from obspy import UTCDateTime

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.header_util import pack_ch_header
from detector.send_receive.client_zmq import ZmqClient

from obspy import *
import numpy as np


def test_receiver(conn_str):
    context = zmq.Context()
    socket = ZmqClient(conn_str, context)

    pyplot.ion()
    figure = pyplot.figure()
    st = Stream()
    check_time = UTCDateTime()
    data_dic = {}
    packet_time = None
    while True:
        size_bytes = socket.recv(4)
        size = int.from_bytes(size_bytes, byteorder='little')
        if not 20 < size < 50000:
            logger.warning('possibly incorrect data size:' + str(size))
            continue
        raw_data = socket.recv(size)
        if not raw_data[:1] == b'{':
            logger.error('no start \'{\' symbol')
            continue
        if raw_data[-1:] != b'}':
            logger.error('incorrect last symbol, \'}\' expected')
            continue
        try:
            json_data = json.loads(raw_data.decode('utf-8'))
        except Exception as e:
            logger.error('cannot parse json data:\n' + str(raw_data) + '\n' + str(e))
            continue
        if 'signal' in json_data:
            sampling_rate = json_data['signal']['sample_rate']
            starttime = UTCDateTime(json_data['signal']['timestmp'])
            if not packet_time:
                packet_time = starttime
            logger.debug('signal received, dt:' + str(starttime))
            chs = json_data['signal']['samples']
            for ch in chs:
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                data = np.frombuffer(bin_signal, dtype='int32')
                if ch in data_dic:
                    data_dic[ch] = np.append(data_dic[ch], data)
                else:
                    data_dic[ch] = data
                #print('data:' + str(data[:100]))
                tr = Trace()
                tr.stats.starttime = starttime
                tr.stats.sampling_rate = sampling_rate
                tr.stats.channel = ch
                tr.data = data
                st += tr
            st.sort().merge()
            st.trim(starttime=st[0].stats.endtime - 10)
            pyplot.clf()
            st.plot(fig=figure)
            pyplot.show()
            pyplot.pause(.1)
            data_dic = {}
        else:
            logger.debug('received packet is not signal')


#test_receiver('tcp://192.168.0.200:5561')
test_receiver('tcp://192.168.0.189:5561')

