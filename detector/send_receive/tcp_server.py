import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class TcpServer:

    def __init__(self, conn_str, context):
        self.identity = None
        self.context = context
        self.conn_str = conn_str
        self.socket = None

    def send(self, data):
        if not self.identity:
            logger.info('open connection')
            self.socket = self.context.socket(zmq.STREAM)
            self.socket.bind(self.conn_str)
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)
            self.identity = self.socket.recv()
            logger.info('id: ' + str(self.identity))
            if len(self.identity) != 5:
                raise Exception('unexpected id len:' + str(len(self.identity)))
            empty_data = self.socket.recv()
            if empty_data:
                raise Exception('empty data expected:' + str(empty_data))
        try:
            self.socket.send(self.identity, zmq.SNDMORE)
            self.socket.send(data)
        except Exception as ex:
            logger.warning('Cannot send the data, supposedly client is closing connection. ' + str(ex))
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
            identity = self.socket.recv()
            logger.info('identity received:' + str(identity) + ' supposed identity:' + str(self.identity))
            empty = self.socket.recv()
            logger.info('empty received? empty data:' + str(empty))
            if empty:
                ex = Exception('empty expected:' + str(empty))
                logger.error(str(ex))
                raise ex
            logger.info('connection is closed')
            self.identity = None
            self.socket.close()

    def __del__(self):
        if self.socket:
            self.socket.close()

