from detector.send_receive.tcp_server import TcpServer


sender_zmq = TcpServer('tcp://*:5555')
sender_zmq.send(b'Have a nice day!')

