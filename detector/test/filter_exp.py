from obspy import *
import numpy as np
from obspy.signal.filter import highpass
from scipy.signal import iirfilter, zpk2sos, sosfilt, sosfilt_zi
#from butterworth import Butter


def bandpass_zi(data, sampling_rate, freqmin, freqmax, zi=None, sos=None):
    corners = 4
    zerophase = False

    fe = 0.5 * sampling_rate
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=sampling_rate, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    if zi is None:
        z, p, k = iirfilter(corners, [low, high], btype='bandpass',
                            ftype='butter', output='zpk')
        sos = zpk2sos(z, p, k)
        zi = sosfilt_zi(sos)
    data, zo = sosfilt(sos, data, zi=zi)
    return data, zo, sos


# st = read()
# st = Stream() + st[0]
# st_filt = st.copy()
# st_filt.filter(type='bandpass', freqmin=.1, freqmax=1)
# st_filt[0].stats.station = 'ND01'
# st_chunked = st[0] / 10
# st_zi = st_chunked.copy()
# for tr in st_chunked:
#     tr.filter(type='bandpass', freqmin=.1, freqmax=1)
#     tr.stats.station = 'ND02'
# zi = None
# sos = None
# for tr in st_zi:
#     tr.data, zi, sos = bandpass_zi(tr.data, tr.stats.sampling_rate, .1, 1, zi, sos)
#     tr.stats.station = 'ND03'
# (st + st_filt + st_chunked.merge() + st_zi.merge()).plot()


