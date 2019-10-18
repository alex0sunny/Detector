import time
import zmq

from obspy import UTCDateTime

from detector.send_receive.server_zmq import ZmqServer

context = zmq.Context()
server = ZmqServer('tcp://*:5555', context)

while True:
    server.send(str(UTCDateTime()).encode())
    time.sleep(1)

