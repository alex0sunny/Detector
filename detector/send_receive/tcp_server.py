import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class TcpServer:

    def __init__(self, conn_str, context):
        self.identity = None
        self.context = context
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.bind(conn_str)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)

    def send(self, data):
        if not self.identity:
            logger.info('open connection')
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
            identity = self.socket.recv()
            empty = self.socket.recv()
            if empty:
                ex = Exception('empty expected:' + str(empty))
                logger.error(str(ex))
                raise ex
            logger.info('connection is closed')
            self.identity = None

    def __del__(self):
        self.socket.close()

