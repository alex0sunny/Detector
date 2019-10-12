from detector.send_receive.sender_zmq import ZmqSender


sender_zmq = ZmqSender('tcp://*:5555')
sender_zmq.send(b'Have a nice day!')

