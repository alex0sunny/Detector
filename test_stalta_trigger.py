import os
import sys
from multiprocessing import Process

from detector.StaLtaTrigger import *

if __name__ == '__main__':
    p_sender = Process(target=sender_test, args=())
    p_receiver = Process(target=sta_lta_picker, args=('ND01', 'EHE', 100, 300, 1, 4, 2, 1))
    p_sender.start()
    p_receiver.start()

