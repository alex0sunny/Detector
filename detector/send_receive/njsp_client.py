import zmq

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.tcp_client import TcpClient


class NjspClient(TcpClient):

    def recv(self, n):
        data = super().recv(n)
        if data[:6] == b'NJSP\0\0':
            logger.info('header received')
            data = data[6:]
            data += super().recv(6)
        return data

