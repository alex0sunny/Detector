import zmq
import time
import logging
from obspy import UTCDateTime

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('')

context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.bind("tcp://*:5555")
#socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.setsockopt(zmq.SNDTIMEO, 2000)
while True:
    logger.debug('open connection')
    net_id = socket.recv()
    empty = socket.recv()
    if empty:
        logger.error('empty expected:' + str(empty))
    logger.debug('connection opened (?)')
    while True:
        try:
            socket.send(net_id, zmq.SNDMORE)
        except:
            logger.warning('cannot send id, supposedly client is closing connection')
            break
        socket.send(str(UTCDateTime()).encode())
        time.sleep(1)
    logger.info('wait for client to close the connection')
    net_id = socket.recv()
    empty = socket.recv()
    if empty:
        logger.error('empty expected:' + str(empty))
    logger.info('client closed the connection')
    time.sleep(1)

