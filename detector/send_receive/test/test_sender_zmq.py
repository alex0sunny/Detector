from detector.send_receive.server_zmq import ZmqServer


sender_zmq = ZmqServer('tcp://*:5555')
sender_zmq.send(b'Have a nice day!')

