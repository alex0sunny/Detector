import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")
socket.connect("tcp://localhost:5565")
while True:
    id = socket.expand()
    message = socket.expand()
    print("received:" + str(message))

