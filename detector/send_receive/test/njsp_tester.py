import logging
import numpy as np
from queue import Queue, Empty
from time import sleep
from matplotlib import pyplot
from obspy import Stream, Trace, UTCDateTime

from detector.send_receive.njsp.njsp import NJSP

show_signal = True

host = 'localhost'
port = 10011
station = 'NDXX'
sample_rate = init_packet = check_time = None

logpath = None
loglevel = logging.DEBUG

format = '%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(level=loglevel, filename=logpath, format=format)
logger = logging.getLogger('router')
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("gpsd").setLevel(logging.WARNING)


njsp = NJSP(logger=logger, log_level=logging.DEBUG)
njsp_params = {
    'reconnect': True,
    'reconnect_period': 10,
    'bson': True,
    'handshake': {
        'subscriptions': ['status', 'log', 'streams', 'alarms'],
        'flush_buffer': False,
        'client_name': 'TRIG'
    }
}

njsp_queue = Queue(100)
reader1 = njsp.add_reader(host, port, 'TRIG', njsp_params, njsp_queue)
print('reader:', reader1)
# reader2 = njsp.add_reader('localhost', 10011, 'TRIG', njsp_params, njsp_queue)

while not njsp.is_alive(reader1):
    logger.info(f'{reader1} connecting...')
    sleep(1)

if show_signal:
    pyplot.ion()
    figure = pyplot.figure()
    st = Stream()

while True:
    #logger.debug(f'qsize:{njsp_queue.qsize()}')
    try:
        packets_data = njsp_queue.get(timeout=1)
    except Empty:
        logger.info('no data')
        continue
    for conn_name, dev_packets in packets_data.items():
        for packet_type, content in dev_packets.items():
            if 'streams' == packet_type and station in content and sample_rate:
                for stream_name, stream_data in content.items():
                    if stream_name == station:
                        starttime = UTCDateTime(stream_data['timestamp'])
                        for ch_name, bytez in stream_data['samples'].items():
                            #stream_data['samples'][ch_name] = len(stream_data['samples'][ch_name])
                            if show_signal:
                                tr = Trace()
                                tr.stats.starttime = starttime
                                tr.stats.sampling_rate = sample_rate
                                tr.stats.channel = ch_name
                                tr.data = np.frombuffer(bytez, 'int')
                                st += tr
                        if not check_time:
                            check_time = starttime
                        if show_signal and starttime > check_time + 1:
                            check_time = starttime
                            st.sort().merge()
                            st.trim(starttime=st[0].stats.endtime - 10)
                            pyplot.clf()
                            st.plot(fig=figure)
                            pyplot.show()
                            pyplot.pause(.1)
                # logger.debug(f'stream content, streams:{list(content.keys())} sample_rate:{sample_rate}')
            if packet_type == 'parameters' and station in content['streams']:
                for stream in list(content['streams'].keys()):
                    if stream != station:
                        del content['streams'][stream]
                init_packet = content
                print('init_packet:', init_packet)
                sample_rate = content['streams'][station]['sample_rate']
        #logger.debug('packets:\n' + str(packets_data))
