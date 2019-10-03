import socket, json, base64
import struct


def listener():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bufsize = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [Before]:%d" % bufsize)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
        bufsize = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [After]:%d" % bufsize)
        s.connect(("192.168.0.200", 10003))

        while True:
            msg_size = struct.unpack('<L', s.recv(4))[0]
            print('msg_size:' + str(msg_size))
            bytes_recvd = 0
            bin_data = b''
            while bytes_recvd < msg_size:
                bin_data += s.recv(msg_size - bytes_recvd)
                bytes_recvd = len(bin_data)
            if bin_data != None:
                #print(str(bin_data))
                last_symb = (bin_data.decode()[-1])
                if last_symb != '}':
                    bin_data = s.recv(10000)
                    last_symb = (bin_data.decode()[-1])
                    print('last symb:' + last_symb)
                    while last_symb != '}':
                        bin_data = s.recv(10000)
                        last_symb = (bin_data.decode()[-1])
                        print('last symb:' + last_symb)
                    #exit(2)
                    continue
                json_data = json.loads(bin_data.decode('utf-8'))
                if 'signal' in json_data:
                    temp = base64.decodebytes(json_data['signal']['samples']["ch1"].encode("ASCII"))
                    #print(base64.decodebytes(json_data['signal']['samples']["ch1"].encode("ASCII")))
            else:
                break
        s.close()


listener()