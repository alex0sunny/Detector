import logging
import time

import numpy as np

from obspy import *
from matplotlib import pyplot

from detector.misc.header_util import chunk_stream, stream_to_dic
from detector.send_receive.njsp.njsp import NJSP
from detector.test.signal_generator import SignalGenerator
import os
import detector.misc as misc
import inspect


logpath = None
loglevel = logging.DEBUG

format = '%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(level=loglevel, filename=logpath, format=format)
logger = logging.getLogger('generator')

njsp = NJSP(logger=logger, log_level=logging.DEBUG)


def send_signal(st, port, units='V'):
    show_signal = False

    signal_generator = SignalGenerator(st)

    #context = zmq.Context()
    if show_signal:
        pyplot.ion()
        figure = pyplot.figure()
    st_vis = Stream()
    check_time = time.time()
    ch_dic = {tr.stats.channel: {'ch_active': True, 'counts_in_volt': tr.stats.k} for tr in st}
    parameters_dic = {
        'parameters': {
            'streams': {
                st[0].stats.station: {
                    'sample_rate': int(st[0].stats.sampling_rate),
                    'channels': ch_dic,
                    'data_format': 'bson'
                }
            }
        }
    }
    streamer_params = {'init_packet': parameters_dic, 'ringbuffer_size': 10}
    streamserver = njsp.add_streamer('', port, streamer_params)

    while True:
        st = signal_generator.get_stream()
        st_add = st.copy()
        for tr_vis in st_add:
            tr_vis.data = np.require(tr_vis.data / tr_vis.stats.k, 'float32')
        st_vis += st_add
        cur_time = time.time()
        if cur_time > check_time + 1:
            check_time = cur_time
            st_vis.sort().merge()
            starttime = st_vis[0].stats.endtime - 5
            st_vis.trim(starttime=starttime)
            if show_signal:
                pyplot.clf()
                st_vis.plot(fig=figure)
                pyplot.show()
                pyplot.pause(.01)
            else:
                time.sleep(.1)  # delete this when return pyplot!
        sts = chunk_stream(st)
        bson_datas = [stream_to_dic(st, units) for st in sts]
        for bson_data in bson_datas:
            njsp.broadcast_data(streamserver, bson_data)
            # data_len = len(bson_data)
            # size_bytes = ('%08x' % data_len).encode()
            # sender.send(size_bytes + bson_data)
            time.sleep(.01)
        #print('broadcasted')
        time.sleep(.1)


base_path = os.path.split(inspect.getfile(misc))[0] + '/'
st = read(base_path + 'st1000.mseed')
for tr in st:
    tr.stats.k = 1000.0
st100 = read(base_path + 'st100.mseed')
for tr in st100:
    tr.stats.k = 10000
    tr.stats.network = 'RU'
    tr.stats.location = '00'
data = st[-1].data
st[-1].data = np.append(data[2000:], data[:2000])

print(st100)
send_signal(st100, 10002)
