# import base64
# import json
#
# from obspy import UTCDateTime
#
# from detector.filter.StaLtaTrigger import logger
# from detector.misc.header_util import pack_ch_header
import base64
import json

import zmq
from obspy import UTCDateTime

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.header_util import pack_ch_header


def signal_receiver(conn_str):
    context = zmq.Context()
    socket = context.socket(zmq.STREAM)
    socket.connect(conn_str)
    id_net = socket.recv()
    socket.recv()   # empty data here

    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:5559')

    while True:
        assert(id_net == socket.recv())
        raw_data = socket.recv()
        size_bytes = raw_data[:4]
        size = int.from_bytes(size_bytes, byteorder='little')
        if not 20 < size < 50000:
            logger.warning('possibly incorrect data size:' + str(size))
            continue
        if not raw_data[4:5] == b'{':
            logger.error('no start \'{\' symbol')
            continue
        while len(raw_data) < size + 4:
            raw_data += socket.recv()
        if len(raw_data) > size + 4:
            logger.error('incorrect data size')
            continue
        if raw_data[-1:] != b'}':
            logger.error('incorrect last symbol, \'}\' expected')
            continue
        try:
            json_data = json.loads(raw_data[4:].decode('utf-8'))
        except Exception as e:
            logger.error('cannot parse json data:\n' + str(raw_data[4:]) + '\n' + str(e))
            continue
        if 'signal' in json_data:
            sampling_rate = json_data['signal']['sample_rate']
            starttime = UTCDateTime(json_data['signal']['timestmp'])
            chs = json_data['signal']['samples']
            for ch in chs:
                bin_header = pack_ch_header('ND01', ch, sampling_rate, starttime._ns)
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                bin_data = bin_header + bin_signal
                socket_pub.send(bin_data)
        else:
            logger.debug('received packet is not signal')
