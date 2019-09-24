from obspy import *


class SignalGenerator:

    def __init__(self, st):
        self.st = st
        starttime = UTCDateTime(UTCDateTime()._get_ns() // 10 ** 9)
        for tr in self.st:
            tr.stats.starttime = starttime
        self.period = st[0].stats.endtime + st[0].stats.delta - st[0].stats.starttime
        self.prev_time = starttime

    def get_stream(self):
        dt = UTCDateTime()
        if self.prev_time < dt < self.st[0].stats.starttime + self.period:
            ret_val = self.st.slice(self.prev_time, dt)
        if dt > self.st[0].stats.starttime + self.period:
            st_sliced = self.st.slice(self.prev_time)
            for tr in self.st:
                tr.stats.starttime = tr.stats.starttime + self.period
            ret_val = (st_sliced + self.st.slice(endtime=dt)).merge()
        self.prev_time = dt
        return ret_val
