#import socket
import zmq
import logging

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('receiver')


context = zmq.Context()

socket = context.socket(zmq.STREAM)
socket.setsockopt(zmq.IDENTITY, b'B')
socket.connect('tcp://192.168.0.200:10003')

logger.debug('identity:' + str(socket.getsockopt(zmq.IDENTITY)))
identity = None
while True:
    if not identity:
        identity = socket.recv()
        logger.debug('identity received')
        empty = socket.recv()
        if empty:
            raise Exception('empty expected:' + str(empty))
        logger.debug('connection established')
    identity_ = socket.recv()
    if identity_ != identity:
        raise Exception('new identity:' + str(identity_))
    logger.debug('identity received')
    data = socket.recv()
    if not data:
        logger.warning('empty received, so try to close on the client side')
        # socket.send(socket.getsockopt(zmq.IDENTITY))
        # socket.send(b'')
        logger.info('supposedly closed on client side')
        identity = None
    else:
        logger.debug('data received:' + str(data))


# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(b'Hello, world')
#     data = s.recv(1024)
#
# print('Received', repr(data))

