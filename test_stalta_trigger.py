from multiprocessing import Process


import os

from detector.filter_trigger.StaLtaTrigger import sta_lta_picker
from detector.filter_trigger.trigger_resender import resend
from detector.send_receive.signal_receiver import signal_receiver

if __name__ == '__main__':
    p_receiver = Process(target=signal_receiver, args=('tcp://192.168.0.189:5555',))
    p_resender = Process(target=resend, args=('tcp://*:5561', ['EHE', 'EHN'], 4, 1))
    p_picker = Process(target=sta_lta_picker, args=('ND01', 'EHE', 100, 300, 1, 4, 2, 1))
    p_picker2 = Process(target=sta_lta_picker, args=('ND01', 'EHN', 100, 300, 1, 4, 2, 1))
    p_picker3 = Process(target=sta_lta_picker, args=('ND01', 'EHZ', 100, 300, 1, 4, 2, 1))
    p_receiver.start()
    p_resender.start()
    p_picker.start()
    p_picker2.start()
    p_picker3.start()

