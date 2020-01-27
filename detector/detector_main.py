from multiprocessing import Process

from detector.filter_trigger.StaLtaTrigger import sta_lta_picker
from detector.filter_trigger.trigger_resender import resend
from detector.misc.globals import Port, CustomThread
from detector.send_receive.signal_receiver import signal_receiver

import zmq


use_thread = False

if __name__ == '__main__':

    context = zmq.Context()
    socket_backend = context.socket(zmq.SUB)
    socket_backend.bind('tcp://*:%d' % Port.backend.value)
    socket_backend.setsockopt(zmq.SUBSCRIBE, b'AP')

    while True:

        kwargs_list = [{'target': signal_receiver,
                        'kwargs': {'conn_str': 'tcp://192.168.0.189:%d' % Port.test_signal.value}},
                       {'target': resend, 'kwargs': {'conn_str': 'tcp://*:%d' % Port.signal_resend.value,
                                                     'triggers': [1, 2], 'pem': 1, 'pet': 1}},
                       {'target': sta_lta_picker,
                        'kwargs': {'trigger_index': 0, 'station': 'ND01', 'channel': 'EHE',
                                   'freqmin': 100, 'freqmax': 300, 'sta': 1, 'lta': 4,
                                   'init_level': 2, 'stop_level': 1}},
                       {'target': sta_lta_picker,
                        'kwargs': {'trigger_index': 1, 'station': 'ND01', 'channel': 'EHN',
                                   'freqmin': 100, 'freqmax': 300, 'sta': 1, 'lta': 4,
                                   'init_level': 2, 'stop_level': 1}},
                       {'target': sta_lta_picker,
                        'kwargs': {'trigger_index': 2, 'station': 'ND01', 'channel': 'EHZ',
                                   'freqmin': 100, 'freqmax': 300, 'sta': 1, 'lta': 4,
                                   'init_level': 2, 'stop_level': 1}}]

        ps = []
        for kwargs in kwargs_list:
            if use_thread:
                p = CustomThread(**kwargs)
            else:
                p = Process(**kwargs)
            ps.append(p)
        for p in ps:
            p.start()

        socket_backend.recv()

        for p in ps:
            p.terminate()
        print('threads stopped')
        continue
    print('after break away from cycle, should exit')
