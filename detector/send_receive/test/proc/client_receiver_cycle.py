import logging

import zmq

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('')

context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.connect("tcp://127.0.0.1:5555")
while True:
    logger.info('try to open connection')
    net_id = socket.recv()
    empty = socket.recv()
    logger.info('connection opened')
    if empty:
        logger.warning('empty expected:' + str(empty))
    while True:
        net_id = socket.recv()
        if len(net_id) != 5:
            logger.error('incorrect id:' + str(net_id))
        message = socket.recv()
        if not message:
            logger.warning('empty message received, consider connection is closed')
            break
        else:
            logger.debug('received:' + str(message))



