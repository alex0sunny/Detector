import time
from multiprocessing import Process

from detector.filter_trigger.StaLtaTrigger import sta_lta_picker
from detector.filter_trigger.trigger_resender import resend
from detector.misc.globals import Port, CustomThread, sources_dic
from backend.trigger_html_util import getTriggerParams
from detector.send_receive.signal_receiver import signal_receiver


import zmq

from detector.send_receive.triggers_proxy import triggers_proxy

use_thread = False

if __name__ == '__main__':

    context = zmq.Context()
    socket_backend = context.socket(zmq.SUB)
    socket_backend.bind('tcp://*:' + str(Port.backend.value))
    socket_backend.setsockopt(zmq.SUBSCRIBE, b'AP')

    while True:

        paramsList = getTriggerParams()

        kwargs_list = []
        for station, conn_data in sources_dic.items():
            kwargs = {'target': signal_receiver,
                      'kwargs': {'conn_str': 'tcp://' + conn_data['address'] + ':' + str(conn_data['port']),
                                 'station_bin': station}}
            kwargs_list.append(kwargs)

        kwargs_list += [{'target': resend, 'kwargs': {'conn_str': 'tcp://*:' + str(Port.signal_resend.value),
                                                      'triggers': [1, 2], 'pem': 1, 'pet': 1}},
                        {'target': triggers_proxy, 'kwargs': {}}]

        for params in paramsList:
            params.update({'station': 'ND01', 'freqmin': 100, 'freqmax': 300, 'init_level': 2, 'stop_level': 1})
            kwargs_list.append({'target': sta_lta_picker, 'kwargs': params})

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
        time.sleep(1)

        for p in ps:
            p.terminate()
        print('threads stopped')
        continue
    print('after break away from cycle, should exit')
