import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)
# 192.168.0.200 10003
socket.bind('tcp://*:5555')
id_sock = socket.recv()
assert not socket.recv()
assert id_sock == socket.recv()
message = socket.recv()
print('received:' + str(message))

