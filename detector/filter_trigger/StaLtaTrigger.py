from _ctypes import sizeof
from io import BytesIO

import numpy as np
from obspy import *
from obspy.signal.trigger import classic_sta_lta

from detector.filter_trigger.RmsTrigger import RmsTrigger, LevelTrigger
from detector.filter_trigger.filter_bandpass import Filter
from detector.filter_trigger.trigger_types import TriggerType
from detector.misc.globals import Port, Subscription, logger
from detector.misc.header_util import prep_name, ChHeader, prep_ch


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
            res1 = self.trigger(data[:self.nsta - 1])
            res2 = self.trigger(data[self.nsta - 1:])
            return np.append(res1, res2)
        if self.buf.size > self.nlta:
            decrement = self.buf[-self.nlta - 1]
        else:
            decrement = 0
        self.buf = self.buf[-self.nlta:]
        self.buf -= decrement
        cum_sum = np.cumsum(data ** 2)
        try:
            next_sta = self.sta + \
                       ((cum_sum + self.buf[-self.nsta - 1]) - self.buf[-self.nsta:-self.nsta + data.size]) / self.nsta
        except Exception as ex:
            logger.debug('buf size:' + str(self.buf.size))
            raise ex
        next_lta = self.lta + (cum_sum - self.buf[-self.nlta:-self.nlta + data.size]) / self.nlta
        self.sta = next_sta[-1]
        self.lta = next_lta[-1]
        self.buf = np.append(self.buf, cum_sum + self.buf[-1])
        # logger.debug('\nsta:' + str(self.sta[-data.size:]))  # + '\nlta:' + str(self.lta[-data.size:]))
        return next_sta / next_lta


class SltaTriggerCore:

    def __init__(self, nsta, nlta):
        if nlta <= nsta:
            raise Exception('incorrect nlta:' + str(nlta) + ' nsta:' + str(nsta))
        self.nsta = nsta
        self.nlta = nlta
        self.buf = np.require(np.zeros(0), dtype='float')
        self.npts = 0

    def trigger(self, data):
        self.buf = np.append(self.buf[-self.nlta:], data)
        if self.buf.size > self.nlta:
            out = classic_sta_lta(self.buf, self.nsta, self.nlta)[-data.size:]
        else:
            out = np.require(np.zeros(data.size), dtype='float')
        return out


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

    def __init__(self, trigger_id, trigger_type, use_filter, freqmin, freqmax,
                 init_level, stop_level, sta=0, lta=0):
        self.trigger_index = trigger_id
        self.trigger_type = trigger_type
        self.use_filter = use_filter
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.init_level = init_level
        self.stop_level = stop_level
        self.trigger_on = False
        self.sta = sta
        self.lta = lta
        self.sample_rate = self.data_trigger = self.filter = None

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        nsta = round(self.sta * self.sample_rate)
        nlta = round(self.lta * self.sample_rate)
        if self.trigger_type == TriggerType.sta_lta:
            self.data_trigger = StaLtaTrigger(nsta, nlta)
        if self.trigger_type == TriggerType.RMS:
            self.data_trigger = RmsTrigger(nsta)
        if self.trigger_type == TriggerType.level:
            self.data_trigger = LevelTrigger()
        if self.use_filter:
            self.filter = Filter(self.sample_rate, self.freqmin, self.freqmax)

    def pick(self, starttime, data):
        triggerings = []
        if not self.sample_rate:
            raise Exception(f'sample rate is not set for trigger {self.trigger_index}')
        if self.filter:
            try:
                data = self.filter.bandpass(data)
            except Exception as ex:
                logger.error(str(ex) + '\ndisable filter')
                self.filter = None
        trigger_data = self.data_trigger.trigger(data)
        if self.init_level >= self.stop_level:
            activ_data = trigger_data > self.init_level
            deactiv_data = trigger_data < self.stop_level
        else:
            activ_data = trigger_data < self.init_level
            deactiv_data = trigger_data > self.stop_level

        date_time = starttime
        # message_start = Subscription.trigger.value + self.trigger_index_s
        while True:
            if self.trigger_on:
                seek_ar = np.where(deactiv_data)[0]
            else:
                seek_ar = np.where(activ_data)[0]
                # logger.debug(f'trigger off, seek_ar:{seek_ar}')
            if seek_ar.size == 0:
                break
            offset = seek_ar[0]
            activ_data = activ_data[offset:]
            deactiv_data = deactiv_data[offset:]
            date_time += offset / self.sample_rate
            self.trigger_on = not self.trigger_on
            triggerings.append([date_time, int(self.trigger_on), self.trigger_index])
            logger.debug(f'triggered, trigger id:{self.trigger_index} on:{self.trigger_on}')
        return triggerings
            #self.socket_trigger.send(message_start + btrig + date_time._ns.to_bytes(8, byteorder='big'))

