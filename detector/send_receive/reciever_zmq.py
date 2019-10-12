import json
import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class ZmqReceiver:

    def __init__(self, conn_str, context):
        self.net_id = 0
        self.context = context
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.connect(conn_str)
        self.net_id = self.socket.recv()
        if len(self.net_id) != 5:
            raise Exception('unexpected id_bytes len:' + str(len(self.net_id)))
        logger.info('id:' + str(self.net_id))
        empty_data = self.socket.recv()
        if empty_data:
            raise Exception('empty data expected:' + str(empty_data))

    def recv(self):
        net_id = self.socket.recv()
        if self.net_id != net_id:
            logger.warning('unexpected id:' + str(net_id) + ' expected:' + str(self.net_id))
        data = self.socket.recv()
        if not data:
            logger.error('empty data received')
        return data

    def __del__(self):
        self.socket.close()

