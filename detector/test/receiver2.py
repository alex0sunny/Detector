import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.bind("tcp://*:5555")
socket.bind("tcp://*:5565")

id = socket.recv()
socket.recv()
id = socket.recv()
message = socket.recv()
print("received:" + str(message))
id = socket.recv()
message = socket.recv()
print("received:" + str(message))
id = socket.recv()
message = socket.recv()
print("received:" + str(message))


