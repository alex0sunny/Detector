from multiprocessing import Process


import os

from detector.filter_trigger.StaLtaTrigger import sta_lta_picker
from detector.filter_trigger.trigger_resender import resend
from detector.send_receive.signal_receiver import signal_receiver

if __name__ == '__main__':
    #p_receiver = Process(target=signal_receiver, args=('tcp://192.168.0.189:5555',))
    p_receiver = Process(target=signal_receiver, args=('tcp://192.168.0.200:10003',))
    p_resender = Process(target=resend, args=('tcp://*:5561', ['ch1'], 3, 1))
    p_picker  = Process(target=sta_lta_picker, args=('ND01', 'ch1', 10, 30, 1, 4, 3.5, 1))
    p_picker2 = Process(target=sta_lta_picker, args=('ND01', 'ch2', 1, 49, 1, 4, 2, 1))
    p_picker3 = Process(target=sta_lta_picker, args=('ND01', 'ch3', 1, 49, 1, 4, 2, 1))
    p_receiver.start()
    p_resender.start()
    p_picker.start()
    p_picker2.start()
    p_picker3.start()

