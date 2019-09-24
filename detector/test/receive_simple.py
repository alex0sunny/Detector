import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")
socket.connect("tcp://localhost:5565")
while True:
    id = socket.recv()
    message = socket.recv()
    print("received:" + str(message))

