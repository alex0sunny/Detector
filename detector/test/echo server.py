import socket

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 5555        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', PORT))
    s.listen(10)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(256)
            print('received data:' + str(data))
            if not data:
                break
            conn.sendall(data)