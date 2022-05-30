# import base64
# import json
#
import base64
import json
import os
#
# from detector.filter.StaLtaTrigger import logger
# from detector.misc.header_util import pack_ch_header
from collections import OrderedDict
from ctypes import cast, POINTER
from time import sleep

import numpy as np
import zmq
from obspy import UTCDateTime

from backend.trigger_html_util import set_source_channels
from detector.filter_trigger.StaLtaTrigger import logger, TriggerWrapper
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import prep_ch, CustomHeader, ChName, ChHeader
from detector.send_receive.njsp.njsp import NJSP_MULTISTREAMREADER

STREAM_IND = 0
STREAM_NAME = None


def signal_receiver(conn_str, station_bin, triggers_params):
    station = station_bin.decode()
    show_signal = False     # os.name == 'nt'
    # if station != 'ND02':
    #     show_signal = False

    context = zmq.Context()
    #socket = NjspClient(conn_str, context)
    str_parts = conn_str.split(':')[-2:]
    host = str_parts[0][2:]
    port = str_parts[-1]
    stream_reader = NJSP_MULTISTREAMREADER()
    stream_reader.add_client(host, port)
    packet = client_name = None
    while not packet or not client_name or 'parameters' not in packet[client_name]:
        try:
            packet = stream_reader.queue.get(timeout=30)
            client_name = list(packet.keys())[0]
        except Exception as ex:
            logger.error('cannot receive init packet:', ex)
            packet = None
    #stream_reader.connected_event.wait()
    #init_packet = stream_reader.init_packet.copy()
    init_packet = packet[client_name]
    STREAM_NAME = list(init_packet['parameters']['streams'].keys())[0]

    socket_pub = context.socket(zmq.PUB)
    conn_str_pub = 'tcp://localhost:' + str(Port.multi.value)
    socket_pub.connect(conn_str_pub)
    socket_buf = context.socket(zmq.PUB)
    socket_buf.connect(conn_str_pub)

    #show_signal = station == 'ND02'
    if station != STREAM_NAME:
        init_packet['parameters']['streams'][station] = \
                init_packet['parameters']['streams'][STREAM_NAME]
        del init_packet['parameters']['streams'][STREAM_NAME]
    params_dic = init_packet['parameters']['streams'][station]

    socket_confirm = context.socket(zmq.SUB)
    socket_confirm.connect('tcp://localhost:' + str(Port.proxy.value))
    socket_confirm.setsockopt(zmq.SUBSCRIBE, Subscription.confirm.value)

    confirmed = None

    trigger_objs_dic = None

    if show_signal:
        from matplotlib import pyplot
        from obspy import Stream, Trace
        pyplot.ion()
        figure = pyplot.figure()
        st = Stream()
    check_time = None
    times_dic = OrderedDict()
    skip_packet = True
    delta_ns = 10 ** 9
    limit_ns = 5 * delta_ns

    chs_ref = []

    #trigger_wrapper = TriggerWrapper(context, 1, TriggerType.sta_lta, 1000, True, 100, 300, 3, 1, 1, 4)
    while True:
        if not stream_reader.alive: #stream_reader.connected_event.is_set():
            logger.warning('kill stream_reader')
            stream_reader.kill()
            #stream_reader = NJSP_STREAMREADER((host, port))
            stream_reader = NJSP_MULTISTREAMREADER()
            stream_reader.add_client(host, port)
            logger.warning('wait connection...')
            # if not stream_reader.connected_event.wait():
            #     continue
            try:
                packet = stream_reader.queue.get(timeout=30)
            except Exception as ex:
                logger.error('cannot receive init packet:', ex)
                packet = None
            if not packet:
                continue
            logger.info('stream_reader connected')
            #init_packet = stream_reader.init_packet.copy()
            client_name = list(packet.keys())[0]
            init_packet = packet[client_name]
            if station != STREAM_NAME:
                init_packet['parameters']['streams'][station] = \
                    init_packet['parameters']['streams'][STREAM_NAME]
                del init_packet['parameters']['streams'][STREAM_NAME]
            socket_buf.send(Subscription.parameters.value + json.dumps(init_packet).encode())
            params_dic = init_packet['parameters']['streams'][station]
            if confirmed:
                socket_buf.send(Subscription.parameters.value + json.dumps(init_packet).encode())
        #logger.debug('wait data packet')
        try:
            packet = stream_reader.queue.get(timeout=0.5)
        except:
            packet = None
        if packet:
            #logger.debug('packet received')
            packet = packet[client_name]
        if not packet or 'streams' not in packet:
            continue
        if not confirmed:
            logger.debug('wait confirmation')
            socket_confirm.recv()
            logger.debug('signal resent confirmed')
            confirmed = True
            socket_buf.send(Subscription.parameters.value + json.dumps(init_packet).encode())
        starttime = UTCDateTime(packet['streams'][STREAM_NAME]['timestamp'])
            #logger.debug('received packet, dt:' + str(starttime))
        if skip_packet:
            times_dic[UTCDateTime()._ns] = starttime._ns
            # print()
            # for tr, ts in times_dic.items():
            #     print(UTCDateTime(tr / (10 ** 9)), ' ', UTCDateTime(ts / (10 ** 9)))
            while len(times_dic.keys()) > 2 and \
                        list(times_dic.keys())[-2] - list(times_dic.keys())[0] > limit_ns:
                del times_dic[list(times_dic.keys())[0]]
            if list(times_dic.keys())[-1] - list(times_dic.keys())[0] > limit_ns and \
                    limit_ns - delta_ns < list(times_dic.values())[-1] - list(times_dic.values())[0] < \
                        limit_ns + 2 * delta_ns:
                skip_packet = False
            else:
                logger.info('skip packet')
                continue
        chs = packet['streams'][STREAM_NAME]['samples']
        if not chs_ref:
            chs_ref = sorted(chs)
            #units = json_data['signal']['counts']
            set_source_channels(station, chs_ref)
        sample_rate = params_dic['sample_rate']
        if trigger_objs_dic is None:
            trigger_objs_dic = {}
            for ch, trigger_params_list in triggers_params.items():
                if ch not in trigger_objs_dic:
                    trigger_objs_dic[ch] = []
                for trigger_params in trigger_params_list:
                    trigger_params['sampling_rate'] = int(sample_rate)
                    trigger_params['context'] = context
                    trigger_objs_dic[ch].append(TriggerWrapper(**trigger_params))
        for ch in chs:
            bin_header = ChHeader(station_bin, ch, int(sample_rate), starttime._ns)
            bin_signal_int = packet['streams'][STREAM_NAME]['samples'][ch]
            k = params_dic['channels'][ch]['counts_in_volt']
            data = np.frombuffer(bin_signal_int, dtype='int32').astype('float') / k
            if ch in trigger_objs_dic:
                for trigger_obj in trigger_objs_dic[ch]:
                    trigger_obj.pick(starttime, data)

            if show_signal:
                tr = Trace()
                tr.stats.starttime = starttime
                tr.stats.sampling_rate = sample_rate
                tr.stats.channel = ch
                tr.data = data
                st += tr

        custom_header = CustomHeader()
        chs_blist = list(map(prep_ch, chs))
        chs_bin = b''.join(chs_blist)
        custom_header.channels = cast(chs_bin, POINTER(ChName * 20)).contents
        custom_header.ns = starttime._ns
        data_dic = packet['streams'][STREAM_NAME]['samples']
        for ch in data_dic:
            ch_data = data_dic[ch]
            data_dic[ch] = base64.encodebytes(ch_data).decode()
        if station != STREAM_NAME:
            packet['streams'][station] = packet['streams'][STREAM_NAME]
            del packet['streams'][STREAM_NAME]
        socket_buf.send(Subscription.signal.value + custom_header +
                            json.dumps(packet).encode())

        if not check_time:
            check_time = starttime
        if show_signal and starttime > check_time + 1:
            check_time = starttime
            st.sort().merge()
            st.trim(starttime=st[0].stats.endtime - 50)
            pyplot.clf()
            st.plot(fig=figure)
            pyplot.show()
            pyplot.pause(.1)

