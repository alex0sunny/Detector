# master
import base64
import json
import socket as sock
import time  # for test only
# from matplotlib import pyplot
from multiprocessing import Process

from obspy import *
import numpy as np
import zmq

import logging

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('detector')
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from detector.header_util import unpack_ch_header, prep_name, pack_ch_header, chunk_stream
from detector.test.filter_exp import bandpass_zi


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
    data_trigger = None
    trigger_on = False
    zi = None
    sos = None
    while True:
        raw_data = socket.recv()
        raw_header = raw_data[8:18]
        #print('raw_header received:' + str(raw_header))
        sampling_rate, starttime = unpack_ch_header(raw_header)
        #print('sampling_rate:' + str(sampling_rate) + ' starttime:' + str(starttime))
        data = np.frombuffer(raw_data[18:], dtype='int32')
        data, zi, sos = bandpass_zi(data, sampling_rate, freqmin, freqmax, zi, sos)
        if not data_trigger:
            nsta = round(sta * sampling_rate)
            nlta = round(lta * sampling_rate)
            data_trigger = StaLtaTrigger(nsta, nlta)
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
                events_list.append({'dt': date_time, 'trigger': True})
                trigger_on = True
            date_time += 1.0 / sampling_rate
        if events_list:
            print('events_list:' + str(events_list))


def sender_test():
    s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    s.setsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF, 8192)
    # s.connect(("192.168.0.200", 10003))
    s.connect(("192.168.0.189", 5555))

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:5559')

    while True:
        size_data = s.recv(4)
        size = int.from_bytes(size_data, byteorder='little')
        if size > 20000:
            print('possibly incorrect size:' + str(size))
        bdata = b''
        bytes_recvd = 0
        while bytes_recvd < size:
            bdata += s.recv(size - bytes_recvd)
            bytes_recvd = len(bdata)
        # print('bdata size:' + str(len(bdata)) + '\nbdata:' + str(bdata))
        if bdata[-1] == 125:
            json_data = json.loads(bdata.decode('utf-8'))
            if 'signal' in json_data:
                sampling_rate = json_data['signal']['sample_rate']
                starttime = UTCDateTime(json_data['signal']['timestmp'])
                #print('current time:' + str(starttime))
                chs = json_data['signal']['samples']
                for ch in chs:
                    bin_header = pack_ch_header('ND01', ch, sampling_rate, starttime._ns)
                    bin_signal = (base64.decodebytes(json_data['signal']['samples'][ch].encode("ASCII")))
                    bin_data = bin_header + bin_signal
                    socket.send(bin_data)
        else:
            while bdata[-1] != 125:
                print('skip packet')
                bdata = s.recv(10000)


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

