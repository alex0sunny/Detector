import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")

id = socket.recv()
socket.recv()
id = socket.recv()
message = socket.recv()
print("received:" + str(message))

