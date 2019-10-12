from obspy import *

from detector.filter_trigger.filter_bandpass import bandpass_zi

st = read()
st = Stream() + st[0]
st_filt = st.copy()
st_filt.filter(type='bandpass', freqmin=.1, freqmax=1)
st_filt[0].stats.station = 'ND01'
st_chunked = st[0] / 10
st_zi = st_chunked.copy()
for tr in st_chunked:
    tr.filter(type='bandpass', freqmin=.1, freqmax=1)
    tr.stats.station = 'ND02'
zi = None
sos = None
for tr in st_zi:
    tr.data, zi, sos = bandpass_zi(tr.data, tr.stats.sampling_rate, .1, 1, zi, sos)
    tr.stats.station = 'ND03'
(st + st_filt + st_chunked.merge() + st_zi.merge()).plot()


