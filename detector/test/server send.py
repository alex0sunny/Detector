import socket
import time

PORT = 5555        # Port to listen on (non-privileged ports are > 1023)

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    conn.sendall(b'smile!')
                    time.sleep(1)
    except ConnectionResetError as ex:
        print(str(ex))
