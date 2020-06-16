# import base64
# import json
#
# from obspy import UTCDateTime
#
# from detector.filter.StaLtaTrigger import logger
# from detector.misc.header_util import pack_ch_header
import base64
import json
from collections import OrderedDict
from ctypes import cast, POINTER
from io import BytesIO
from time import sleep

import zmq
import numpy as np
from matplotlib import pyplot
from obspy import UTCDateTime, Stream, Trace

from backend.trigger_html_util import set_source_channels
from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import prep_ch, CustomHeader, ChName, ChHeader
from detector.send_receive.njsp_client import NjspClient

STREAM_IND = 0
STREAM_NAME = None


def signal_receiver(conn_str, station_bin=b'ND01'):
    show_signal = True

    context = zmq.Context()
    socket = NjspClient(conn_str, context)

    if show_signal:
        pyplot.ion()
        figure = pyplot.figure()
    st = Stream()
    check_time = None

    chs_ref = []

    params_dic = None
    # while True:
    #     header = socket.recv(6)
    #     if header == b'NJSP\0\0':
    #         print('header received')
    #         break
    while True:
        size_bytes = socket.recv(8)
        size = int(size_bytes.decode(), 16)
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
            json_data = json.loads(raw_data.decode('utf-8'), object_pairs_hook=OrderedDict)
        except Exception as e:
            logger.error('cannot parse json data:\n' + str(raw_data) + '\n' + str(e))
            continue
        if 'parameters' in json_data:
            logger.debug('received parameters')
            streams_dic = json_data['parameters']['streams']
            STREAM_NAME = list(streams_dic.keys())[STREAM_IND]
            params_dic = streams_dic[STREAM_NAME]
            #print('params bytes sent to inner socket:' + str(Subscription.parameters.value + size_bytes + raw_data))
        if 'streams' in json_data:
            #sampling_rate = json_data['streams']['sample_rate']
            starttime = UTCDateTime(json_data['streams'][STREAM_NAME]['timestamp'])
            logger.debug('received packet, dt:' + str(starttime))
            chs = json_data['streams'][STREAM_NAME]['samples']
            if not chs_ref:
                chs_ref = sorted(chs)
                set_source_channels(station_bin.decode(), chs_ref)
            for ch in chs:
                sample_rate = params_dic['sample_rate']
                bin_signal_int = (base64.decodebytes(json_data['streams'][STREAM_NAME]['samples'][ch].encode("ASCII")))
                k = params_dic['channels'][ch]['counts_in_volt']

                data = np.frombuffer(bin_signal_int, dtype='int32').astype('float32') / k
                tr = Trace()
                tr.stats.starttime = starttime
                tr.stats.sampling_rate = sample_rate
                tr.stats.channel = ch
                tr.data = data
                st += tr

            if not check_time:
                check_time = starttime
            if show_signal and starttime > check_time + 1:
                check_time = starttime
                st.sort().merge()
                st.trim(starttime=st[0].stats.endtime - 10)
                pyplot.clf()
                st.plot(fig=figure)
                pyplot.show()
                pyplot.pause(.1)


signal_receiver('tcp://localhost:5561')

