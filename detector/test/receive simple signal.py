import zmq
from obspy import *

from detector.misc.header_util import bin_to_stream

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")
while True:
    id = socket.expand()
    bin_data = socket.expand()
    if not bin_data:
        print('empty data')
    else:
        print('bin_data size:' + str(len(bin_data)))
        st = bin_to_stream(bin_data)
        print("received:" + str(st))

