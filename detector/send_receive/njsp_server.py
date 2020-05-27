import zmq

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.tcp_server import TcpServer


class NjspServer(TcpServer):

    def __init__(self, params_bstr, conn_str, context):
        super().__init__(conn_str, context)
        self.params_bstr = params_bstr

    def send(self, data):
        if not self.identity:
            print('send params')
            super().send(self.params_bstr)
        super().send(data)

