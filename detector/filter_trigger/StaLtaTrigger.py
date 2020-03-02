import base64
import json
import socket as sock
# from matplotlib import pyplot

from obspy import *
import numpy as np
import zmq

import logging

from detector.misc.globals import Port
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


def sta_lta_picker(ind, station, channel, freqmin, freqmax, sta, lta, init_level, stop_level):
    trigger_index_s = ('%02d' % ind).encode()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://localhost:%d' % Port.signal_route.value)
    topicfilter = prep_name(station).decode() + prep_name(channel).decode()
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    socket_trigger = context.socket(zmq.PUB)
    socket_trigger.connect('tcp://localhost:%d' % Port.trigger.value)
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
                socket_trigger.send(b'ND01' + trigger_index_s + b'0' +
                                    date_time._ns.to_bytes(8, byteorder='big'))
                logger.debug('detriggered, ch:' + channel)
                # events_list.append({'channel': channel, 'dt': date_time, 'trigger': False})
                trigger_on = False
            if not trigger_on and a:
                socket_trigger.send(b'ND01' + trigger_index_s + b'1' +
                                    date_time._ns.to_bytes(8, byteorder='big'))
                logger.debug('triggered, ch:' + channel)
                #events_list.append({'channel': channel, 'dt': date_time, 'trigger': True})
                trigger_on = True
            date_time += 1.0 / sampling_rate
        # if events_list:
        #     print('events_list:' + str(events_list))