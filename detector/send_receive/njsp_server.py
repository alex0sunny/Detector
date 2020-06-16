import zmq

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.tcp_server import TcpServer


class NjspServer(TcpServer):

    def __init__(self, conn_str, context):
        super().__init__(conn_str, context)
        self.params_bstr = None

    def set_params(self, params_bstr):
        self.params_bstr = params_bstr

    def send(self, data):
        if not self.params_bstr:
            raise Exception('no params bytes')
        if not self.identity:
            logger.info('send params')
            super().send(b'NJSP\0\0');
            super().send(self.params_bstr)
        super().send(data)

