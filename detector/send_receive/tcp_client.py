import json
import zmq

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import CONNECTION_TOUT


class TcpClient:

    def __init__(self, conn_str, context):
        self.conn_str = conn_str
        self.identity = None
        self.socket = None
        self.buf = b''
        self.context = context
        self.__open__()

    def __open__(self):
        if self.socket:
            raise Exception('socket is already created')
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.connect(self.conn_str)
        if not self.socket.poll(CONNECTION_TOUT):
            raise ConnectionException('timeout on socket openning')
        self.identity = self.socket.recv()
        assert(len(self.identity) == 5)
        assert(self.socket.recv() == b'')
        self.buf = b''

    def __close__(self):
        if self.socket is None:
            raise Exception('socket is already closed or was not created')
        if self.identity:
            self.socket.send(self.identity, zmq.SNDMORE)
            self.socket.send(b'')
        self.identity = None
        self.socket.close()
        self.socket = None

    def recv(self, n):
        while len(self.buf) < n:
            self.buf += self.__recv__()
        data = self.buf[:n]
        self.buf = self.buf[n:]
        return data

    def __recv__(self):
        while True:
            try:
                if self.socket is None:
                    self.__open__()
                if not self.socket.poll(CONNECTION_TOUT):
                    raise ConnectionException('tout while receiving data')
                assert (self.identity == self.socket.recv())
                data = self.socket.recv()
                if not data:
                    raise ConnectionException('connection was closed by the server')
                return data
            except ConnectionException as ex:
                logger.warning(ex)
                if self.socket:
                    self.__close__()

    def __del__(self):
        if self.socket:
            self.socket.close()


class ConnectionException(Exception):
    pass

