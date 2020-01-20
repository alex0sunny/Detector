import zmq
import time
from obspy import *

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://alexeynote:5556")

while True:
    socket.send(str(UTCDateTime()).encode())
    time.sleep(1)

