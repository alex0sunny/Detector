import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")
id = socket.getsockopt(zmq.IDENTITY)
socket.send(id, zmq.SNDMORE)
socket.send(b"message")

