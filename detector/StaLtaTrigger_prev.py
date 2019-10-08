# master
import base64
import json
import socket as sock
import time     # for test only
#from matplotlib import pyplot
from multiprocessing import Process

from obspy import *
import numpy as np
import zmq

from detector.header_util import unpack_ch_header, prep_name, pack_ch_header, chunk_stream
from detector.test.filter_exp import bandpass_zi
# from detector.test.signal_generator import SignalGenerator

import logging

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


class StaLtaTriggerCore:

    def __init__(self, nsta, nlta):
        self.nsta = nsta
        self.nlta = nlta
        #print('nlta:' + str(nlta))
        self.lta = np.require(np.zeros(nlta), dtype='float32')
        self.sta = np.require(np.zeros(nsta), dtype='float32')
        self.buf = self.lta.copy()

    def trigger(self, data):
        self.buf = self.buf[-self.nlta:]
        self.sta = self.sta[-self.nsta:]
        self.lta = self.lta[-self.nlta:]
        for data_val in data ** 2:
            next_sta = self.sta[-1] + (data_val - self.buf[-self.nsta]) / self.nsta
            logger.debug('next sta:' + str(next_sta))
            self.sta = np.append(self.sta, next_sta)
            next_lta = self.lta[-1] + (data_val - self.buf[0]) / self.nlta
            #logger.debug('next lta:' + str(next_lta))
            self.lta = np.append(self.lta, next_lta)
            self.buf = np.append(self.buf, data_val)[1:]
        return self.sta[-data.size:] / self.lta[-data.size:]


nsta = 5
nlta = 10
data = np.arange(20)
slTrigger = StaLtaTriggerCore(nsta, nlta)
slTrigger.trigger(data)

