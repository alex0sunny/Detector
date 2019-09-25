from obspy import *
import numpy as np
import zmq

from detector.header_util import unpack_ch_header, prep_name


class StaLtaTriggerCore:

    def __init__(self, nsta, nlta):
        self.nsta = nsta
        self.nlta = nlta
        self.lta = np.require(np.zeros(nlta), dtype='float32')
        self.sta = np.require(np.zeros(nsta), dtype='float32')
        self.buf = self.lta.copy()

    def trigger(self, data):
        for data_val in data ** 2:
            next_sta = self.sta[-1] + (data_val - self.buf[-self.nsta]) / self.nsta
            self.sta = np.append(self.sta, next_sta)
            next_lta = self.lta[-1] + (data_val - self.buf[0]) / self.nlta
            self.lta = np.append(self.lta, next_lta)
            self.buf = np.append(self.buf, data_val)[1:]
        return self.sta[-data.size:] / self.lta[-data.size:]


class StaLtaTrigger:

    def __init__(self, nsta, nlta):
        self.triggerCore = StaLtaTriggerCore(nsta, nlta)
        self.bufsize = 0

    def trigger(self, data):
        retVal = self.triggerCore.trigger(data)
        self.bufsize += data.size
        tail = self.bufsize - self.triggerCore.nlta + 1
        if 0 < tail < data.size:
            retVal = np.append(np.require(np.zeros(data.size - tail), dtype='float32'), retVal[-tail:])
        elif tail <= 0:
            retVal = np.require(np.zeros(data.size), dtype='float32')
        return retVal


def sta_lta_picker(station, channel, sta, lta, init_level, stop_level):
    port = 5559
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:%s" % port)
    topicfilter = prep_name(station) + prep_name(channel)
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    data_trigger = None
    trigger_on = False
    while True:
        raw_data = socket.recv()
        raw_header = raw_data[:12]
        sampling_rate, starttime = unpack_ch_header(raw_header)
        data = np.frombuffer(raw_data[12:], dtype='float32')
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
                events_list.append({'dt:': date_time, 'trigger': True})
                trigger_on = True
            date_time += 1.0 / sampling_rate


st = read()
tr = st[1]
sta = 1
lta = 4
nsta = int(tr.stats.sampling_rate * sta)
nlta = int(tr.stats.sampling_rate * lta)
tr_triggered = tr.copy()
data = tr.data
slTrigger = StaLtaTrigger(nsta, nlta)
# tr.data = slTrigger.trigger(data)
data_trigger = np.empty(0, 'float32')
for tr in tr / 10:
    data = slTrigger.trigger(tr.data)
    data_trigger = np.append(data_trigger, data)
print('data_trigger size:' + str(data_trigger.size))
tr_triggered.data = data_trigger
tr_triggered.plot()
