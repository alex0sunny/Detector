import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.bind("tcp://*:5555")
socket.bind("tcp://*:5565")

id = socket.expand()
socket.expand()
id = socket.expand()
message = socket.expand()
print("received:" + str(message))
id = socket.expand()
message = socket.expand()
print("received:" + str(message))
id = socket.expand()
message = socket.expand()
print("received:" + str(message))


