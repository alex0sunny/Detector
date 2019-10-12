import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)
# 192.168.0.200 10003
socket.bind("tcp://*:5555")

id = socket.recv()
print('id:' + str(id) + ' type:' + str(type(id)))
socket.recv()

socket.send(id, zmq.SNDMORE)
socket.send(b"message")

