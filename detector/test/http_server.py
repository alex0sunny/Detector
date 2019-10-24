import zmq
import random
import time

context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.bind("tcp://*:8080")
while True:
    id = socket.recv()
    print('id:' + str(id))
    raw = []
    while len(raw) == 0 or len(raw) == len(id):
        raw = socket.recv()
        print('raw:' + str(raw))
    http_response = 'HTTP/1.0 200 OK\r\n' + \
        'HTTP/1.0 200 OK\r\n' + \
        'Content-Type: text/plain\r\n' + \
        '\r\n' + \
        'Hello, World!'
    socket.send(id, zmq.SNDMORE)
    socket.send_string(http_response, zmq.SNDMORE)
    socket.send(id, zmq.SNDMORE)
    socket.send(b'', zmq.SNDMORE)


