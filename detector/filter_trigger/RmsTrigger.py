# master

from obspy import *
import numpy as np
import zmq

from detector.misc.header_util import unpack_ch_header, prep_name, prep_ch
from detector.filter_trigger import bandpass_zi


class RmsTriggerCore:

    def __init__(self, n, threshold):
        self.n = n
        self.threshold = threshold
        self.buf = np.require(np.zeros(n), dtype='float32')
        self.rms = self.buf.copy()

    def trigger(self, data):
        self.rms = self.rms[-self.n:]
        self.buf = self.buf[-self.n:]
        for data_val in data ** 2:
            rms_val = (self.sta[-1] + (data_val - self.buf[-self.n]) / self.n)
            self.rms = np.append(self.rms, rms_val)
            self.buf = np.append(self.buf, data_val)
        return self.rms[-data.size:] ** .5


class RmsTrigger:

    def __init__(self, n):
        self.triggerCore = RmsTriggerCore(n)
        self.bufsize = 0

    def trigger(self, data):
        ret_val = self.triggerCore.trigger(data)
        self.bufsize += data.size
        tail = self.bufsize - self.triggerCore.n + 1
        if 0 < tail < data.size:
            ret_val = np.append(np.require(np.zeros(data.size - tail), dtype='float32'), ret_val[-tail:])
        elif tail <= 0:
            ret_val = np.require(np.zeros(data.size), dtype='float32')
        return ret_val


def rms_picker(station, channel, freqmin, freqmax, len, init_level, stop_level):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://localhost:5559')
    topicfilter = prep_name(station) + prep_ch(channel)
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
    data_trigger = None
    trigger_on = False
    zi = None
    while True:
        raw_data = socket.expand()
        raw_header = raw_data[7:17]
        # print('raw_header received:' + str(raw_header))
        sampling_rate, starttime = unpack_ch_header(raw_header)
        # print('sampling_rate:' + str(sampling_rate) + ' starttime:' + str(starttime))
        data = np.frombuffer(raw_data[17:], dtype='int32')
        data, zi = bandpass_zi(data, sampling_rate, freqmin, freqmax, zi)
        if not data_trigger:
            n = round(len * sampling_rate)
            data_trigger = RmsTrigger(n)
        trigger_data = data_trigger.trigger(data)
        activ_data = trigger_data > init_level
        deactiv_data = trigger_data < stop_level
        date_time = starttime
        events_list = []
        for a, d in zip(activ_data, deactiv_data):
            if trigger_on and d:
                events_list.append({'dt': date_time, 'trigger': False})
                trigger_on = False
            if not trigger_on and a:
                events_list.append({'dt:': date_time, 'trigger': True})
                trigger_on = True
            date_time += 1.0 / sampling_rate
        if events_list:
            print('events_list:' + str(events_list))


# st = read()
# tr = st[1]
# sta = 1
# lta = 4
# nsta = int(tr.stats.sampling_rate * sta)
# nlta = int(tr.stats.sampling_rate * lta)
# tr_triggered = tr.copy()
# data = tr.data
# slTrigger = StaLtaTrigger(nsta, nlta)
# # tr.data = slTrigger.trigger(data)
# data_trigger = np.empty(0, 'float32')
# for tr in tr / 10:
#     data = slTrigger.trigger(tr.data)
#     data_trigger = np.append(data_trigger, data)
# print('data_trigger size:' + str(data_trigger.size))
# tr_triggered.data = data_trigger
# tr_triggered.plot()