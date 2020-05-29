import json
import time
from collections import OrderedDict
from multiprocessing import Process

import numpy as np

from obspy import *
from matplotlib import pyplot

#from detector.misc.globals import ports_map
from backend.trigger_html_util import getSources
from detector.misc.header_util import chunk_stream, stream_to_json
from detector.send_receive.njsp_server import NjspServer
from detector.test.signal_generator import SignalGenerator
from detector.send_receive.tcp_server import TcpServer
import zmq
import os
import detector.misc as misc
import inspect


def send_signal(st, conn_str, units='V'):
    signal_generator = SignalGenerator(st)

    context = zmq.Context()
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
                    'channels': ch_dic
                }
            }
        }
    }
    json_str = json.dumps(OrderedDict(parameters_dic))
    size_bytes = ('%08x' % len(json_str)).encode()
    sender = NjspServer(size_bytes + json_str.encode(), conn_str, context)

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
            pyplot.clf()
            st_vis.plot(fig=figure)
            pyplot.show()
            pyplot.pause(.01)
        sts = chunk_stream(st)
        json_datas = [stream_to_json(st, units).encode('utf8') for st in sts]
        for json_data in json_datas:
            data_len = len(json_data)
            # print('bdata size:' + str(data_len))
            #size_bytes = int(data_len).to_bytes(4, byteorder='little')
            size_bytes = ('%08x' % data_len).encode()
            sender.send(size_bytes + json_data)
            time.sleep(.01)
        time.sleep(.1)


base_path = os.path.split(inspect.getfile(misc))[0] + '/'
st = read(base_path + 'st1000.mseed')
for tr in st:
    tr.stats.k = 1000.0
st100 = read(base_path + 'st100.mseed')
for tr in st100:
    tr.stats.k = 10000
data = st[-1].data
st[-1].data = np.append(data[2000:], data[:2000])
sources_dic = getSources()
print(sources_dic)
stations = list(sources_dic.keys())
kwargs_list = [{'target': send_signal,
                'kwargs': {'st': st,
                           'conn_str': 'tcp://*:' + str(sources_dic[stations[0]]['port'])}},
               {'target': send_signal,
                'kwargs': {'st': st100[:3], 'units': 'A',
                           'conn_str': 'tcp://*:' + str(sources_dic[stations[1]]['port'])}}]
if __name__ == '__main__':
    for kwargs in kwargs_list:
        Process(**kwargs).start()

