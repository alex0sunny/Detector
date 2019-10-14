from obspy import *
import numpy as np
from obspy.signal.filter import highpass
from scipy.signal import iirfilter, zpk2sos, sosfilt, sosfilt_zi


# from butterworth import Butter


class Filter:

    def __init__(self, sampling_rate, freqmin, freqmax):
        self.sampling_rate = sampling_rate
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.zi = None
        self.sos = None
        self.corners = 4
        self.zerophase = False
        self.fe = 0.5 * sampling_rate
        self.low = freqmin / self.fe
        self.high = freqmax / self.fe

    def bandpass(self, data):
        # raise for some bad scenarios
        if self.high - 1.0 > -1e-6:
            msg = ("Selected high corner frequency ({}) of bandpass is at or "
                   "above Nyquist ({}). Applying a high-pass instead.").format(
                self.freqmax, self.fe)
            warnings.warn(msg)
            return highpass(data, freq=self.freqmin, df=self.sampling_rate, corners=self.corners,
                            zerophase=self.zerophase)
        if self.low > 1:
            msg = "Selected low corner frequency is above Nyquist."
            raise ValueError(msg)
        if self.zi is None:
            z, p, k = iirfilter(self.corners, [self.low, self.high], btype='bandpass',
                                ftype='butter', output='zpk')
            self.sos = zpk2sos(z, p, k)
            self.zi = sosfilt_zi(self.sos)
        data, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return data
