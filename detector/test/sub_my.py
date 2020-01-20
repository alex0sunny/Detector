import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind("tcp://*:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    socket.recv()
    print('received')

