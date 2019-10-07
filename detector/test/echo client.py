import socket

HOST = '192.168.0.189'  # The server's hostname or IP address
PORT = 5555        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, world')
    #mes, = s.recv()
    data = s.recv(1024)

print('Received', repr(data))