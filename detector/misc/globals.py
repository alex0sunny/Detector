# ports_map = {'signal': 10003, 'test_signal': 5555, 'signal_route': 5559, 'internal_resend': 5560,
#              'signal_resend': 5561, 'trigger': 5562, 'backend': 5563}
import ctypes
from enum import Enum
import threading
from threading import Thread

import logging
from time import sleep

import zmq

import os

if os.name == 'nt':
    logpath = None  # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/trigger.log'
else:
    logpath = '/media/sdcard/logs/trigger/trigger.log'
loglevel = logging.DEBUG

format = '%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(level=loglevel, filename=logpath, format=format)
logger = logging.getLogger('globals')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("gpsd").setLevel(logging.WARNING)

CONNECTION_TOUT = 5000


ENCODING = 'cp1250'


class Port(Enum):
    signal = 10003
    signal_route = 5559
    internal_resend = 5560
    signal_resend = 10010
    multi = 5562
    proxy = 5563
    backend = 5564


# sources_dic = {'ND01': {'host': 'localhost', 'port': 5555},
#                'ND02': {'host': 'localhost', 'port': 5565}}


class Subscription(Enum):
    signal  = bytes([1])
    intern  = bytes([2])
    trigger = bytes([3])
    raw     = bytes([4])
    rule    = bytes([5])
    test    = bytes([6])
    parameters = bytes([7])
    confirm = bytes([8])


class CustomThread(Thread):

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for thread_id, thread in threading._active.items():
            if thread is self:
                return thread_id

    def raise_exception(self, ex=None):
        thread_id = self.get_id()
        logger.debug('stop thread ' + str(thread_id))
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(ex))
        # if res > 1:
        #     ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        #     logger.debug('Exception raise failure')

    def terminate(self):
        self.raise_exception(SystemExit)
        sleep(.1)
        while self.is_alive():
            logger.warning('fail to abort thread ' + str(self.get_id()))
            logger.info('raise system error')
            self.raise_exception(SystemError)
            sleep(.1)
            logger.info('raise Exception')
            self.raise_exception(Exception)
            sleep(.1)
            logger.info('raise ZMQError')
            self.raise_exception(zmq.ZMQError)
            sleep(.1)
        self.join()


action_names_dic0 = {1: 'relayA', 2: 'relayB', 3: 'seedlk'}

