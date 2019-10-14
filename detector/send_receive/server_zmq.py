import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class ZmqServer:

    def __init__(self, conn_str, context):
        self.net_id = None
        self.context = context
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.bind(conn_str)
        self.net_id = self.socket.recv()
        logger.info('id: ' + str(self.net_id))
        if len(self.net_id) != 5:
            raise Exception('unexpected id len:' + str(len(self.net_id)))
        empty_data = self.socket.recv()
        if empty_data:
            raise Exception('empty data expected:' + str(empty_data))

    def send(self, data):
        self.socket.send(self.net_id, zmq.SNDMORE)
        self.socket.send(data)

    def __del__(self):
        self.socket.close()

