# import base64
# import json
#
# from obspy import UTCDateTime
#
# from detector.filter.StaLtaTrigger import logger
# from detector.misc.header_util import pack_ch_header
import base64
import json
from ctypes import cast, POINTER
from io import BytesIO

import zmq
from obspy import UTCDateTime

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, Subscription, channelsUpdater
from detector.misc.header_util import prep_ch, CustomHeader, ChName, ChHeader
from detector.send_receive.tcp_client import TcpClient


def signal_receiver(conn_str, station_bin):
    context = zmq.Context()
    socket = TcpClient(conn_str, context)

    socket_pub = context.socket(zmq.PUB)
    conn_str_pub = 'tcp://localhost:' + str(Port.multi.value)
    socket_pub.connect(conn_str_pub)
    socket_buf = context.socket(zmq.PUB)
    socket_buf.connect(conn_str_pub)

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
                #bin_header = pack_ch_header(station_bin, ch, sampling_rate, starttime._ns)
                bin_header = ChHeader(station_bin, ch, int(sampling_rate), starttime._ns)
                bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                bin_data = BytesIO(bin_header).read() + bin_signal
                socket_pub.send(Subscription.intern.value + bin_data)
            #chs_bin = len(chs).to_bytes(1, byteorder='big') + b''.join(list(map(prep_ch, chs)))
            #custom_header = (ns_bin + chs_bin).ljust(50)
            custom_header = CustomHeader()
            chs_blist = list(map(prep_ch, chs))
            channelsUpdater.update(station_bin, chs_blist)
            chs_bin = b''.join(chs_blist)
            custom_header.channels = cast(chs_bin, POINTER(ChName * 20)).contents
            custom_header.ns = starttime._ns
            #logger.debug('chs_bin:' + str(chs_bin))
            socket_buf.send(Subscription.signal.value + custom_header + size_bytes + raw_data)

