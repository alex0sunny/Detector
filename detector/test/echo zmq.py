#import socket
import zmq

context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.connect("tcp://localhost:5555")

id = socket.getsockopt(zmq.IDENTITY)
print('id:' + str(id))
while True:
    socket.send(id, zmq.SNDMORE)
    print('id sent')
    socket.send(b'Hello, world') #, zmq.SNDMORE)
    print('message sent')

    id = socket.expand()
    print('received id:' + str(id))
    data = socket.expand()
    print('received data:' + str(data))




# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(b'Hello, world')
#     data = s.recv(1024)
#
# print('Received', repr(data))

