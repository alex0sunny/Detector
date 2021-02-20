# import base64
# import json
#
# from obspy import UTCDateTime
#
# from detector.filter.StaLtaTrigger import logger
# from detector.misc.header_util import pack_ch_header
import base64
import json
import os
import socket
from collections import OrderedDict
from ctypes import cast, POINTER

import numpy as np
from obspy import UTCDateTime, Stream, Trace

from backend.trigger_html_util import set_source_channels
from detector.filter_trigger.StaLtaTrigger import logger, TriggerWrapper
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import prep_ch, CustomHeader, ChName, ChHeader

STREAM_IND = 0
STREAM_NAME = None


def signal_receiver(host, port):

    st = Stream()
    skip_packet = True
    delta_ns = 10 ** 9
    limit_ns = 5 * delta_ns

    chs_ref = []

    params_dic = None

    #trigger_wrapper = TriggerWrapper(context, 1, TriggerType.sta_lta, 1000, True, 100, 300, 3, 1, 1, 4)
    while True:
        try:
            socket_raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_address = (host, port)
            socket_raw.connect(server_address)
            socket_wrapper = SocketWrapper(socket_raw)
            header = socket_wrapper.recv(6)
            if header != b'NJSP\0\0':
                mes = f'incorrect header:{header}'
                logger.error(mes)
                raise Exception(mes)
            logger.info('header received')
            while True:
                size_bytes = socket_wrapper.recv(8)
                if len(size_bytes) != 8:
                    logger.error(f'incorrect len: {len(size_bytes)}, expected:8')
                size = int(size_bytes.decode(), 16)
                if not 20 < size < 50000:
                    logger.warning('possibly incorrect data size:' + str(size))
                    continue
                raw_data = socket_wrapper.recv(size)
                if len(raw_data) != size:
                    logger.error(f'incorrect len: {len(raw_data)}, expected:{size}')
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
                        #units = json_data['signal']['counts']
                    sample_rate = params_dic['sample_rate']

        except Exception as ex:
            logger.error(f'exception:{ex}')
        finally:
            socket_raw.close()


class SocketWrapper:
    def __init__(self, socket_raw):
        self.socket_raw = socket_raw
        self.buf = b''

    def recv(self, n):
        data = self.buf
        while len(data) < n:
            data += self.socket_raw.recv(n - len(data))
        self.buf = data[n:]
        return data[:n]

signal_receiver('192.168.0.226', 10001)