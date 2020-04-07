import numpy as np
from obspy import *


class RmsTrigger:

    def __init__(self, n):
        self.n = n
        self.v = .0
        self.buf = np.require(np.zeros(n), dtype='float32')

    def trigger(self, data):
        if data.size >= self.n:
            res1 = self.trigger(data[:self.n - 1])
            res2 = self.trigger(data[self.n - 1:])
            return np.append(res1, res2)
        if self.buf.size > self.n:
            decrement = self.buf[-self.n - 1]
        else:
            decrement = 0
        self.buf = self.buf[-self.n:]
        self.buf -= decrement
        cum_sum = np.cumsum(data.astype('float32')**2)
        next_vals = self.v + (cum_sum - self.buf[-self.n:-self.n + data.size]) / self.n
        self.v = next_vals[-1]
        self.buf = np.append(self.buf, cum_sum+self.buf[-1])
        #logger.debug('\nsta:' + str(self.sta[-data.size:]))  # + '\nlta:' + str(self.lta[-data.size:]))
        return np.sqrt(next_vals)


# class RmsTrigger:
#
#     def __init__(self, n):
#         self.triggerCore = RmsTriggerCore(n)
#         self.bufsize = 0
#
#     def trigger(self, data):
#         ret_val = self.triggerCore.trigger(data)
#         self.bufsize += data.size
#         tail = self.bufsize - self.triggerCore.nlta + 1
#         if 0 < tail < data.size:
#             ret_val = np.append(np.require(np.zeros(data.size - tail), dtype='float32'), ret_val[-tail:])
#         elif tail <= 0:
#             ret_val = np.require(np.zeros(data.size), dtype='float32')
#         return ret_val

# tr = read()[0]
# data = tr.data
# tr_trig = tr.copy()
# tr_trig.stats.station = 'trig'
# trigger_core = RmsTriggerCore(100)
# tr_trig.data = trigger_core.trigger(data)
# tr_tg02 = tr.copy()
# tr_tg02.stats.station = 'tg02'
# trigger_core = RmsTriggerCore(1000)
# tr_tg02.data = np.full(tr_tg02.stats.npts, 100., 'float32')
# tr_tg02.data = trigger_core.trigger(tr_tg02.data)
#
# (Stream() + tr_tg02 + tr_trig).plot()
# print(tr_tg02.data[:10])
# print(tr_tg02.data[10:20])
# print(tr_tg02.data[20:])
# print(tr_tg02.data[900:1000])
#
