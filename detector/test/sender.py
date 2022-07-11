import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect('tcp://localhost:5555')
id_sock = socket.getsockopt(zmq.IDENTITY)
socket.send(id_sock, zmq.SNDMORE)
socket.send(b'message')
