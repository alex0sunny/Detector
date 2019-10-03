from obspy import *
import numpy as np


def prep_name(stch):
    if type(stch) == bytes:
        stch = stch.decode()
    stch = stch.strip()[-4:].ljust(4)
    return stch.encode()


def pack_ch_header(station, channel, sampling_rate, stamp_ns):
    bin_data = prep_name(station) + prep_name(channel)
    bin_data += int(sampling_rate).to_bytes(2, byteorder='big')
    bin_data += stamp_ns.to_bytes(8, byteorder='big')
    return bin_data


def unpack_ch_header(bin_data):
    sampling_rate = int.from_bytes(bin_data[:2], byteorder='big')
    stamp_ns = int.from_bytes(bin_data[2:10], byteorder='big')
    stamp = UTCDateTime(stamp_ns / 10 ** 9)
    return sampling_rate, stamp


def pack_header(station, sampling_rate, stamp_ns, mask_string):
    if len(station) != 4:
        print('incorrect station len:' + str(len(station)))
        exit(1)
    if type(station) != bytes:
        station = station.encode()
    bin_data = station
    bin_data += int(sampling_rate).to_bytes(2, byteorder='big')
    bin_data += (4).to_bytes(2, byteorder='big')
    bin_data += stamp_ns.to_bytes(8, byteorder='big')
    bin_data += int(mask_string.encode(), 8).to_bytes(8, byteorder='big')
    return bin_data


def unpack_header(bin_data):
    if len(bin_data) != 24:
        print('incorrect header length:' + str(len(bin_data)))
        exit(1)
    station = bin_data[:4].decode()
    sampling_rate = int.from_bytes(bin_data[4:6], byteorder='big')
    capacity = int.from_bytes(bin_data[6:8], byteorder='big')
    if capacity != 4:
        print('unexpected capacity:' + str(capacity))
        exit(1)
    stamp_ns = int.from_bytes(bin_data[8:16], byteorder='big')
    chans_mask = int.from_bytes(bin_data[16:24], byteorder='big')
    stamp = UTCDateTime(stamp_ns / 10**9)
    n_of_chans = bin(chans_mask).count('1')
    return station, sampling_rate, stamp, n_of_chans


def chunk_stream(st):
    stats = st[0].stats
    npts = stats.npts
    n_of_chans = len(st)
    k = max(1, npts * n_of_chans * 4 // 1000)
    sts = [Stream(trs) for trs in zip(*[tr / k for tr in st]) if trs[0]]    # bug walkaround?
    for st in sts:
        if not st:
            print('error:' + str(st))
            exit(1)
    return sts


def stream_to_bin(st):
    stats = st[0].stats
    station = stats.station
    sampling_rate = stats.sampling_rate
    stamp_ns = stats.starttime._ns
    n_of_chans = len(st)
    mask_string = '1' * n_of_chans
    bin_stats = pack_header(station, sampling_rate, stamp_ns, mask_string)
    data_multiplexed = np.asarray([tr.data for tr in st]).flatten(order='F')
    return bin_stats + data_multiplexed.tobytes()


def bin_to_stream(bin_data):
    station, sampling_rate, stamp, n_of_chans = unpack_header(bin_data[:24])
    data_multiplexed = np.frombuffer(bin_data[24:], dtype='int32')
    print('n_of_chans:' + str(n_of_chans) + ' data size:' + str(data_multiplexed.shape))
    data2d = data_multiplexed.reshape(n_of_chans, -1, order='F')
    st = Stream()
    for i in range(n_of_chans):
        tr = Trace(data=data2d[i])
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = stamp
        tr.stats.channel = 'ch' + str(i+1)
        tr.stats.station = station
        st += tr
    return st


# st = read('D:/converter_data/example/onem.mseed')
# sts = chunk_stream(st)
# for st in sts:
#     print('chunked st:' + str(st))
#     if not st[0]:
#         print('Empty stream!')
#         exit(1)

# st.plot()
# print('st:' + str(st))
# print('st to bin')
# bdata = stream_to_bin(st)
# print('bin to st')
# st = bin_to_stream(bdata)
# print('st:' + str(st))
# st.plot()

