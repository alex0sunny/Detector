import base64
import json

import zmq
from obspy import UTCDateTime

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.header_util import pack_ch_header
from detector.send_receive.client_zmq import ZmqClient


def test_receiver(conn_str):
    context = zmq.Context()
    socket = ZmqClient(conn_str, context)

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
            logger.debug('signal received, dt:' + str(starttime))
            chs = json_data['signal']['samples']
            for ch in chs:
                bin_header = pack_ch_header('ND01', ch, sampling_rate, starttime._ns)
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                bin_data = bin_header + bin_signal
        else:
            logger.debug('received packet is not signal')


test_receiver('tcp://192.168.0.189:5561')

