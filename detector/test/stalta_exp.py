from obspy import *
from obspy.signal.trigger import classic_sta_lta_py
import numpy as np

def offline_sta_lta(data, nsta, nlta, buf = None):
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(data ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    if not buf:
        buf = np.array([])
    np.append(buf, data)
    return sta, lta, buf


def custom_sta_lta(data, nsta, nlta, buf=None, sta=None, lta=None):
    if buf == None:
        buf = np.require([], dtype='float32')
    if buf.size < nlta:
        buf.resize(nlta)
    if buf.size > nlta + nsta:
        buf = buf[-nlta:]
    new_vals = np.cumsum(data ** 2).astype('float32')
    for i in range(len(new_vals)):
        if buf.size + i >= nsta:
            subtrahend = buf[-nsta + i]
            divider = nsta
        else:
            subtrahend = 0
            divider = buf.size + i
        sta_val = sta[-1] + (new_vals[i] - subtrahend) / divider
        sta = np.append(sta, sta_val)

        if buf.size + i >= nlta:
            subtrahend = buf[-nlta + i]
            divider = nlta
        else:
            subtrahend = 0
            divider = buf.size + i
        lta_val = sta[-1] + (new_vals[i] - subtrahend) / divider
        lta = np.append(lta, lta_val)

    return sta, lta, np.append(buf, new_vals)


