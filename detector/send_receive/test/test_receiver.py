from obspy import UTCDateTime
import base64
import json
from collections import OrderedDict
import zmq
import numpy as np

from backend.trigger_html_util import set_source_channels
from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.njsp_client import NjspClient

STREAM_IND = 0
STREAM_NAME = None


def test_receiver(conn_str):
    show_signal = True

    context = zmq.Context()
    socket = NjspClient(conn_str, context)

    if show_signal:
        from matplotlib import pyplot
        from obspy import Stream, Trace
        pyplot.ion()
        figure = pyplot.figure()
        st = Stream()
    check_time = None
    times_dic = OrderedDict()
    skip_packet = True
    delta_ns = 10 ** 9
    limit_ns = 5 * delta_ns

    chs_ref = []

    params_dic = None

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
        if 'streams' in json_data:
            starttime = UTCDateTime(json_data['streams'][STREAM_NAME]['timestamp'])
            if skip_packet:
                times_dic[UTCDateTime()._ns] = starttime._ns
                while len(times_dic.keys()) > 2 and \
                        list(times_dic.keys())[-2] - list(times_dic.keys())[0] > limit_ns:
                    del times_dic[list(times_dic.keys())[0]]
                if list(times_dic.keys())[-1] - list(times_dic.keys())[0] > limit_ns and \
                        limit_ns - delta_ns < list(times_dic.values())[-1] - list(times_dic.values())[0] < \
                        limit_ns + 2 * delta_ns:
                    skip_packet = False
                else:
                    logger.info('skip packet')
                    continue
            chs = json_data['streams'][STREAM_NAME]['samples']
            if not chs_ref:
                chs_ref = sorted(chs)
                #set_source_channels(station_bin.decode(), chs_ref)
            sample_rate = params_dic['sample_rate']
            for ch in chs:
                bin_signal_int = (base64.decodebytes(json_data['streams'][STREAM_NAME]['samples'][ch].encode("ASCII")))
                k = params_dic['channels'][ch]['counts_in_volt']
                data = np.frombuffer(bin_signal_int, dtype='int32').astype('float') / k

                if show_signal:
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


test_receiver('tcp://192.168.0.225:10001')

