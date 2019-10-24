from multiprocessing import Process
from threading import Thread

from detector.filter_trigger.StaLtaTrigger import sta_lta_picker
from detector.filter_trigger.trigger_resender import resend
from detector.send_receive.signal_receiver import signal_receiver

use_thread = True

if __name__ == '__main__':

    # p_receiver = Process(target=signal_receiver, args=('tcp://192.168.0.200:10003',))

    # p_receiver = Process(target=signal_receiver, kwargs={'conn_str': 'tcp://192.168.0.200:10003'})
    # p_resender = Process(target=resend, args=('tcp://*:5561', ['ch1'], 1, 1))
    # p_picker = Process(target=sta_lta_picker, args=('ND01', 'ch1', 10, 30, 1, 4, 3, 1))
    # p_picker2 = Process(target=sta_lta_picker, args=('ND01', 'ch2', 10, 30, 1, 4, 2, 1))
    # p_picker3 = Process(target=sta_lta_picker, args=('ND01', 'ch3', 10, 30, 1, 4, 2, 1))
    kwargs_list = [{'target': signal_receiver, 'kwargs': {'conn_str': 'tcp://192.168.0.200:10003'}},
                   {'target': resend, 'kwargs': {'conn_str': 'tcp://*:5561', 'channels': ['ch1'], 'pem': 1, 'pet': 1}},
                   {'target': sta_lta_picker,
                    'kwargs': {'station': 'ND01', 'channel': 'ch1', 'freqmin': 10, 'freqmax': 30, 'sta': 1, 'lta': 4,
                               'init_level': 3, 'stop_level': 1}},
                   {'target': sta_lta_picker,
                    'kwargs': {'station': 'ND01', 'channel': 'ch2', 'freqmin': 10, 'freqmax': 30, 'sta': 1, 'lta': 4,
                               'init_level': 3, 'stop_level': 1}},
                   {'target': sta_lta_picker,
                    'kwargs': {'station': 'ND01', 'channel': 'ch3', 'freqmin': 10, 'freqmax': 30, 'sta': 1, 'lta': 4,
                               'init_level': 3, 'stop_level': 1}}]
    #Process(**kwargs_list[0]).start()

    for kwargs in kwargs_list:
        if use_thread:
            Thread(**kwargs).start()
        else:
            Process(**kwargs).start()

    # p_receiver.start()
    # p_resender.start()
    # p_picker.start()
    # p_picker2.start()
    # p_picker3.start()

