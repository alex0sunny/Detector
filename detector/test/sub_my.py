from time import sleep

import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.CONFLATE, 1)
socket.bind("tcp://*:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    mes = socket.recv()
    print('received' + mes.decode())
    sleep(1.5)

