import time
import zmq

from obspy import UTCDateTime

from detector.send_receive.tcp_server import TcpServer

context = zmq.Context()
server = TcpServer('tcp://*:5555', context)

while True:
    server.send(str(UTCDateTime()).encode())
    time.sleep(1)

