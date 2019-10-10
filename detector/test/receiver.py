import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)
# 192.168.0.200 10003
socket.bind("tcp://*:5555")
id = socket.expand()
socket.expand()
id = socket.expand()
message = socket.expand()
print("received:" + str(message))

