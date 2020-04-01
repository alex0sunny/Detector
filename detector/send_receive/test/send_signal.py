import time
from multiprocessing import Process

import numpy as np

from obspy import *
from matplotlib import pyplot

#from detector.misc.globals import ports_map
from detector.misc.globals import Port, sources_dic
from detector.misc.header_util import chunk_stream, stream_to_json
from detector.test.signal_generator import SignalGenerator
from detector.send_receive.tcp_server import TcpServer
import zmq
import os
import detector.misc as misc
import inspect


def send_signal(st, conn_str):
    signal_generator = SignalGenerator(st)

    context = zmq.Context()
    sender = TcpServer(conn_str, context)
    pyplot.ion()
    figure = pyplot.figure()
    st_vis = Stream()
    check_time = time.time()
    while True:
        st = signal_generator.get_stream()
        st_vis += st.copy()
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
        json_datas = [stream_to_json(st).encode('utf8') for st in sts]
        for json_data in json_datas:
            data_len = len(json_data)
            # print('bdata size:' + str(data_len))
            size_bytes = int(data_len).to_bytes(4, byteorder='little')
            sender.send(size_bytes + json_data)
            time.sleep(.01)
        time.sleep(.1)


base_path = os.path.split(inspect.getfile(misc))[0] + '/'
st = read(base_path + 'st1000.mseed')
st100 = read(base_path + 'st100.mseed')
data = st[-1].data
st[-1].data = np.append(data[2000:], data[:2000])

kwargs_list = [{'target': send_signal,
                'kwargs': {'st': st,
                           'conn_str': 'tcp://*:' + str(sources_dic['ND01']['port'])}},
               {'target': send_signal,
                'kwargs': {'st': st100,
                           'conn_str': 'tcp://*:' + str(sources_dic['ND02']['port'])}}]
if __name__ == '__main__':
    for kwargs in kwargs_list:
        Process(**kwargs).start()

