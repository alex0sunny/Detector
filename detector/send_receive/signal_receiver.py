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

import zmq
import numpy as np
from obspy import UTCDateTime

from backend.trigger_html_util import set_source_channels
from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import prep_ch, CustomHeader, ChName, ChHeader
from detector.send_receive.tcp_client import TcpClient

STREAM_IND = 0
STREAM_NAME = None


def signal_receiver(conn_str, station_bin):
    context = zmq.Context()
    socket = TcpClient(conn_str, context)

    socket_pub = context.socket(zmq.PUB)
    conn_str_pub = 'tcp://localhost:' + str(Port.multi.value)
    socket_pub.connect(conn_str_pub)
    socket_buf = context.socket(zmq.PUB)
    socket_buf.connect(conn_str_pub)

    chs_ref = []

    params_dic = None
    while True:
        header = socket.recv(6)
        if header == b'NJSP\0\0':
            print('header received')
            break
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
            print('received parameters')
            streams_dic = json_data['parameters']['streams']
            STREAM_NAME = list(streams_dic.keys())[STREAM_IND]
            params_dic = streams_dic[STREAM_NAME]
        if 'streams' in json_data:
            #sampling_rate = json_data['streams']['sample_rate']
            starttime = UTCDateTime(json_data['streams'][STREAM_NAME]['timestamp'])
            chs = json_data['streams'][STREAM_NAME]['samples']
            if not chs_ref:
                chs_ref = sorted(chs)
                #units = json_data['signal']['counts']
                set_source_channels(station_bin.decode(), chs_ref)
            for ch in chs:
                #bin_header = pack_ch_header(station_bin, ch, sampling_rate, starttime._ns)
                sample_rate = params_dic['sample_rate']
                bin_header = ChHeader(station_bin, ch, int(sample_rate), starttime._ns)
                bin_signal_int = (base64.decodebytes(json_data['streams'][STREAM_NAME]['samples'][ch].encode("ASCII")))
                # test_signal = np.frombuffer(bin_signal_int, dtype='int32').astype('float32') / k
                # if np.max(test_signal) > 1:
                #     logger.info('exceed 1:\n' + str(test_signal))
                #logger.info('sampling_rate:' + str(sampling_rate) + ' k:' + str(k))
                k = params_dic['channels'][ch]['counts_in_volt']
                bin_signal = (np.frombuffer(bin_signal_int, dtype='int32').astype('float32') / k).tobytes()
                bin_data = BytesIO(bin_header).read() + bin_signal
                socket_pub.send(Subscription.intern.value + bin_data)
            custom_header = CustomHeader()
            chs_blist = list(map(prep_ch, chs))
            chs_bin = b''.join(chs_blist)
            custom_header.channels = cast(chs_bin, POINTER(ChName * 20)).contents
            custom_header.ns = starttime._ns
            socket_buf.send(Subscription.signal.value + custom_header + size_bytes + raw_data)

