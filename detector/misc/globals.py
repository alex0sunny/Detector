# ports_map = {'signal': 10003, 'test_signal': 5555, 'signal_route': 5559, 'internal_resend': 5560,
#              'signal_resend': 5561, 'trigger': 5562, 'backend': 5563}
import ctypes
from enum import Enum
import threading
from threading import Thread

import logging

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('globals')

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Port(Enum):
    signal = 10003
    signal_route = 5559
    internal_resend = 5560
    signal_resend = 10001
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


class CustomThread(Thread):

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for thread_id, thread in threading._active.items():
            if thread is self:
                return thread_id

    def raise_exception(self):
        thread_id = self.get_id()
        logger.debug('stop thread ' + str(thread_id))
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            logger.debug('Exception raise failure')

    def terminate(self):
        self.raise_exception()
        self.join()

