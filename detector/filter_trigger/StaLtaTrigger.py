# master
import base64
import json
import socket as sock
# from matplotlib import pyplot

from obspy import *
import numpy as np
import zmq

import logging
from detector.misc.header_util import unpack_ch_header, prep_name, pack_ch_header
from detector.filter_trigger.filter_bandpass import Filter

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('detector')
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# from detector.test.signal_generator import SignalGenerator


class StaLtaTriggerCore:

    def __init__(self, nsta, nlta):
        self.nsta = nsta
        self.nlta = nlta
        # print('nlta:' + str(nlta))
        self.lta = .0
        self.sta = .0
        self.buf = np.require(np.zeros(nlta), dtype='float32')

    def trigger(self, data):
        if data.size >= self.nsta:
            res1 = self.trigger(data[:self.nsta-1])
            res2 = self.trigger(data[self.nsta-1:])
            return np.append(res1, res2)
        if self.buf.size > self.nlta:
            decrement = self.buf[-self.nlta - 1]
        else:
            decrement = 0
        self.buf = self.buf[-self.nlta:]
        self.buf -= decrement
        cum_sum = np.cumsum(data.astype('float32')**2)
        next_sta = self.sta + \
            (cum_sum - self.buf[-self.nsta:-self.nsta + data.size] + self.buf[-self.nsta - 1]) / self.nsta
        next_lta = self.lta + (cum_sum - self.buf[-self.nlta:-self.nlta + data.size]) / self.nlta
        self.sta = next_sta[-1]
        self.lta = next_lta[-1]
        self.buf = np.append(self.buf, cum_sum+self.buf[-1])
        #logger.debug('\nsta:' + str(self.sta[-data.size:]))  # + '\nlta:' + str(self.lta[-data.size:]))
        return next_sta / next_lta


class StaLtaTrigger:

    def __init__(self, nsta, nlta):
        self.triggerCore = StaLtaTriggerCore(nsta, nlta)
        self.bufsize = 0

    def trigger(self, data):
        ret_val = self.triggerCore.trigger(data)
        self.bufsize += data.size
        tail = self.bufsize - self.triggerCore.nlta + 1
        if 0 < tail < data.size:
            ret_val = np.append(np.require(np.zeros(data.size - tail), dtype='float32'), ret_val[-tail:])
        elif tail <= 0:
            ret_val = np.require(np.zeros(data.size), dtype='float32')
        return ret_val


def sta_lta_picker(station, channel, freqmin, freqmax, sta, lta, init_level, stop_level):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://localhost:5559')
    topicfilter = prep_name(station).decode() + prep_name(channel).decode()
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    socket_events = context.socket(zmq.PUB)
    socket_events.connect('tcp://localhost:5562')
    # events_list = []

    data_trigger = None
    trigger_on = False
    filter = None
    while True:
        raw_data = socket.recv()
        raw_header = raw_data[8:18]
        #print('raw_header received:' + str(raw_header))
        sampling_rate, starttime = unpack_ch_header(raw_header)
        #print('sampling_rate:' + str(sampling_rate) + ' starttime:' + str(starttime))
        data = np.frombuffer(raw_data[18:], dtype='int32')
        if not filter:
            filter = Filter(sampling_rate, freqmin, freqmax)
        data = filter.bandpass(data)
        if not data_trigger:
            nsta = round(sta * sampling_rate)
            nlta = round(lta * sampling_rate)
            data_trigger = StaLtaTrigger(nsta, nlta)
        trigger_data = data_trigger.trigger(data)
        activ_data = trigger_data > init_level
        deactiv_data = trigger_data < stop_level
        date_time = starttime
        #events_list = []
        for a, d in zip(activ_data, deactiv_data):
            if trigger_on and d:
                socket_events.send(b'ND01' + channel.encode() + b'0')
                # events_list.append({'channel': channel, 'dt': date_time, 'trigger': False})
                trigger_on = False
            if not trigger_on and a:
                socket_events.send(b'ND01' + channel.encode() + b'1')
                #events_list.append({'channel': channel, 'dt': date_time, 'trigger': True})
                trigger_on = True
            date_time += 1.0 / sampling_rate
        # if events_list:
        #     print('events_list:' + str(events_list))


# st = read('D:/converter_data/example/onem.mseed')
# tr = st[0]
# sta = 2
# lta = 10
# nsta = int(tr.stats.sampling_rate * sta)
# nlta = int(tr.stats.sampling_rate * lta)
# tr_triggered = tr.copy()
# tr_triggered.stats.station = 'tri'
# data = tr.data
# slTrigger = StaLtaTrigger(nsta, nlta)
# # tr.data = slTrigger.trigger(data)
# data_trigger = np.empty(0, 'float32')
# for tr_chunked in tr / 10:
#     data = slTrigger.trigger(tr_chunked.data)
#     data_trigger = np.append(data_trigger, data)
# # logger.debug('data_trigger size:' + str(data_trigger.size))
# tr_triggered.data = data_trigger
# tr_classic = tr.copy()
# tr_classic.trigger(type='classicstalta', sta=sta, lta=lta)
# tr_classic.stats.station = 'cla'
# (Stream() + tr + tr_triggered + tr_classic).plot(equal_scale=False)


# nsta = 1000
# nlta = 3000
# data = np.arange(10000)
# slTrigger = StaLtaTriggerCore(nsta, nlta)
# data_trigger = slTrigger.trigger(data)
# i = 1000
# while i < 3000:
#     logger.debug('\ndata_trigger:' + str(data_trigger[i:i+100]))
#     i += 100

