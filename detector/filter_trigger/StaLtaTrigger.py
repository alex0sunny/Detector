from _ctypes import sizeof
from io import BytesIO
from time import sleep

from obspy import *
import numpy as np
import zmq

import logging

from obspy.signal.trigger import classic_sta_lta

from detector.filter_trigger.RmsTrigger import RmsTrigger, LevelTrigger
from detector.filter_trigger.trigger_types import TriggerType
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import prep_name, ChHeader, prep_ch
from detector.filter_trigger.filter_bandpass import Filter

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('detector')

logging.getLogger("matplotlib").setLevel(logging.WARNING)


# from detector.test.signal_generator import SignalGenerator


class StaLtaTriggerCore:

    def __init__(self, nsta, nlta):
        if nlta <= nsta:
            raise Exception('incorrect nlta:' + str(nlta) + ' nsta:' + str(nsta))
        self.nsta = nsta
        self.nlta = nlta
        # print('nlta:' + str(nlta))
        self.lta = .0
        self.sta = .0
        self.buf = np.require(np.zeros(nlta), dtype='float')
        logger.debug('buf size:' + str(self.buf.size))

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
        cum_sum = np.cumsum(data**2)
        try:
            next_sta = self.sta + \
                ((cum_sum + self.buf[-self.nsta - 1]) - self.buf[-self.nsta:-self.nsta + data.size]) / self.nsta
        except Exception as ex:
            logger.debug('buf size:' + str(self.buf.size))
            raise ex
        next_lta = self.lta + (cum_sum - self.buf[-self.nlta:-self.nlta + data.size]) / self.nlta
        self.sta = next_sta[-1]
        self.lta = next_lta[-1]
        self.buf = np.append(self.buf, cum_sum+self.buf[-1])
        #logger.debug('\nsta:' + str(self.sta[-data.size:]))  # + '\nlta:' + str(self.lta[-data.size:]))
        return next_sta / next_lta


class SltaTriggerCore:

    def __init__(self, nsta, nlta):
        if nlta <= nsta:
            raise Exception('incorrect nlta:' + str(nlta) + ' nsta:' + str(nsta))
        self.nsta = nsta
        self.nlta = nlta
        self.buf = np.require(np.zeros(nlta), dtype='float')

    def trigger(self, data):
        self.buf = np.append(self.buf[-self.nlta:], data)
        return classic_sta_lta(self.buf, self.nsta, self.nlta)[-data.size:]


class StaLtaTrigger:

    def __init__(self, nsta, nlta):
        self.triggerCore = SltaTriggerCore(nsta, nlta)
        self.bufsize = 0

    def trigger(self, data):
        ret_val = self.triggerCore.trigger(data)
        self.bufsize += data.size
        tail = self.bufsize - self.triggerCore.nlta + 1
        if 0 < tail < data.size:
            ret_val = np.append(np.require(np.zeros(data.size - tail), dtype='float'), ret_val[-tail:])
        elif tail <= 0:
            ret_val = np.require(np.zeros(data.size), dtype='float')
        return ret_val


class TriggerWrapper:

    def __init__(self, context, trigger_id, trigger_type, sampling_rate, use_filter, freqmin, freqmax,
                 init_level, stop_level, sta=0, lta=0):
        self.trigger_index_s = ('%02d' % trigger_id).encode()
        self.trigger_type = trigger_type
        self.sampling_rate = sampling_rate
        self.init_level = init_level
        self.stop_level = stop_level
        self.trigger_on = False
        if use_filter:
            self.filter = Filter(sampling_rate, freqmin, freqmax)
        else:
            self.filter = None
        nsta = round(sta * sampling_rate)
        nlta = round(lta * sampling_rate)
        self.data_trigger = None
        if trigger_type == TriggerType.sta_lta:
            self.data_trigger = StaLtaTrigger(nsta, nlta)
        if trigger_type == TriggerType.RMS:
            self.data_trigger = RmsTrigger(nsta)
        if trigger_type == TriggerType.level:
            self.data_trigger = LevelTrigger()
        self.socket_trigger = context.socket(zmq.PUB)
        self.socket_trigger.connect('tcp://localhost:' + str(Port.multi.value))

    def pick(self, starttime, data):
        if self.filter:
            data = self.filter.bandpass(data)
        trigger_data = self.data_trigger.trigger(data)
        if self.init_level >= self.stop_level:
            activ_data = trigger_data > self.init_level
            deactiv_data = trigger_data < self.stop_level
        else:
            activ_data = trigger_data < self.init_level
            deactiv_data = trigger_data > self.stop_level
        date_time = starttime
        message_start = Subscription.trigger.value + self.trigger_index_s
        for a, d in zip(activ_data, deactiv_data):
            if self.trigger_on and d:
                self.socket_trigger.send(message_start + b'0' + date_time._ns.to_bytes(8, byteorder='big'))
                self.socket_trigger.send(message_start + b'0' + date_time._ns.to_bytes(8, byteorder='big'))
                logger.debug('detriggered, trigger id:' + str(self.trigger_index_s))
                self.trigger_on = False
            if not self.trigger_on and a:
                self.socket_trigger.send(message_start + b'1' + date_time._ns.to_bytes(8, byteorder='big'))
                self.socket_trigger.send(message_start + b'1' + date_time._ns.to_bytes(8, byteorder='big'))
                logger.debug('triggered, trigger id:' + str(self.trigger_index_s))
                self.trigger_on = True
            date_time += 1.0 / self.sampling_rate



def trigger_picker(ind, station, channel, trigger_type, use_filter, freqmin, freqmax, init_level, stop_level,
                   sta, lta=0):
    #trigger_index_s = ('%02d' % ind).encode()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://localhost:' + str(Port.proxy.value))
    station_bin = prep_name(station)
    socket.setsockopt(zmq.SUBSCRIBE, Subscription.intern.value + station_bin + prep_ch(channel))

    #socket_trigger = context.socket(zmq.PUB)
    #socket_trigger.connect('tcp://localhost:' + str(Port.multi.value))

    #data_trigger = None
    #trigger_on = False
    #filter = None
    trigger_wrapper = None
    while True:
        #sleep(.1)
        raw_data = socket.recv()[1:]
        header = ChHeader()
        header_size = sizeof(ChHeader)
        BytesIO(raw_data[:header_size]).readinto(header)
        sampling_rate = header.sampling_rate
        starttime = UTCDateTime(header.ns / 10 ** 9)
        data = np.frombuffer(raw_data[header_size:], dtype='float')
        if not trigger_wrapper:
            trigger_wrapper = TriggerWrapper(context, ind, trigger_type, sampling_rate, use_filter, freqmin,
                                             freqmax, init_level, stop_level, sta, lta)
        trigger_wrapper.pick(starttime, data)

