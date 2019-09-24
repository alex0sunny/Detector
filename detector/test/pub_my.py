import zmq
import random
import time
import pickle

from obspy import *

context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.connect("tcp://localhost:5556")
id = socket.getsockopt(zmq.IDENTITY)
print('id:' + str(id))

# st = read()
# for tr in st:
#     tr.data = tr.data[:100]

while True:
    socket.send(id, zmq.SNDMORE)
    socket.send_string(str(UTCDateTime()), zmq.SNDMORE)
    # socket.send(id, zmq.SNDMORE)
    # socket.send(b'', zmq.SNDMORE)
    time.sleep(10)


