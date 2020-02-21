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
from detector.misc.globals import Port
from detector.misc.header_util import pack_ch_header, prep_name
from detector.send_receive.tcp_client import TcpClient


def signal_receiver(conn_str):
    context = zmq.Context()
    socket = TcpClient(conn_str, context)

    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:%d' % Port.signal_route.value)
    socket_buf = context.socket(zmq.PUB)
    socket_buf.bind('tcp://*:%d' % Port.internal_resend.value)

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
            chs = json_data['signal']['samples']
            for ch in chs:
                bin_header = pack_ch_header('ND01', ch, sampling_rate, starttime._ns)
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                bin_data = bin_header + bin_signal
                socket_pub.send(bin_data)
            ns_bin = int.to_bytes(starttime._ns, 8, byteorder='big')
            chs_bin = len(chs).to_bytes(1, byteorder='big') + b''.join(list(map(prep_name, chs)))
            custom_header = (b'head' + ns_bin + chs_bin).ljust(50)
            #logger.debug('chs_bin:' + str(chs_bin))
            socket_buf.send(custom_header)
            socket_buf.send(size_bytes + raw_data)

