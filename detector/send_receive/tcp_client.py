import json
import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class TcpClient:

    def __init__(self, conn_str, context):
        self.identity = None
        self.buf = b''
        self.context = context
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.connect(conn_str)

    def recv(self, n):
        while len(self.buf) < n:
            self.buf += self.__recv__()
        data = self.buf[:n]
        self.buf = self.buf[n:]
        return data

    def __recv__(self):
        while True:
            if not self.identity:
                logger.info('try to open connection')
                self.identity = self.socket.recv()
                if len(self.identity) != 5:
                    raise Exception('unexpected id len:' + str(self.identity))
                empty = self.socket.recv()
                if empty:
                    raise Exception('empty data expected:' + str(empty))
                logger.info('connection is opened, id:' + str(self.identity))
            self.socket.recv()
            data = self.socket.recv()
            if data:
                break
            logger.warning('empty message received, consider connection is closed')
            self.identity = None
            self.buf = b''
        return data

    def __del__(self):
        self.socket.close()

