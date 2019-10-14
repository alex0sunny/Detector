import base64

from matplotlib import pyplot
from obspy import *
import numpy as np
import time

from detector.send_receive.client_zmq import NdasReceiver
from detector.filter_trigger.StaLtaTrigger import logger

pyplot.ion()
figure = pyplot.figure()

cur_time = time.time()
st = Stream()

receiver = NdasReceiver('tcp://192.168.0.200:10003')
while True:
    json_data = receiver.recv()
    if 'signal' in json_data:
        # print(json_data)
        sampling_rate = json_data['signal']['sample_rate']
        starttime = UTCDateTime(json_data['signal']['timestmp'])
        for ch in json_data['signal']['samples']:
            bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
            # print('bin signal received')
            data = np.frombuffer(bin_signal, dtype='int32')
            # print('data:' + str(data[:100]))
            tr = Trace()
            tr.stats.starttime = starttime
            tr.stats.sampling_rate = sampling_rate
            tr.stats.channel = ch
            tr.data = data
            st += tr
        if time.time() > cur_time + 2:
            logger.debug('bufsize:' + str(len(receiver.buffer_manager.buf)))
            st.sort()
            st.merge(fill_value='latest')
            st = st.trim(starttime=st[-1].stats.endtime - 5)
            cur_time = time.time()
            pyplot.clf()
            st.plot(fig=figure)
            pyplot.show()
            pyplot.pause(.1)
