import json
import zmq

from detector.filter_trigger.StaLtaTrigger import logger


class ZmqReceiver:

    def __init__(self, conn_str):
        self.net_id = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.STREAM)
        self.socket.connect(conn_str)
        id_bytes = self.socket.recv()
        logger.debug('id bytes: ' + str(id_bytes))
        if len(id_bytes) != 5:
            raise Exception('unexpected id_bytes len:' + str(len(id_bytes)))
        self.net_id = int.from_bytes(id_bytes, byteorder='big')
        logger.info('id:' + str(self.net_id))
        empty_data = self.socket.recv()
        if empty_data:
            raise Exception('empty data expected:' + str(empty_data))

    def recv(self):
        id_bytes = self.socket.recv()
        if len(id_bytes) != 5:
            logger.error('unexpected id_bytes len:' + str(len(id_bytes)))
            return id_bytes
        id_ = int.from_bytes(id_bytes, byteorder='big')
        if self.net_id != id_:
            logger.warning('unexpected id:' + str(id_))
        data = self.socket.recv()
        if not data:
            logger.error('empty data received')
        return data

    def __del__(self):
        self.socket.close()
        self.context.destroy()


class ReceiverBufferManager:

    def __init__(self, conn_str):
        self.receiver = ZmqReceiver(conn_str)
        self.buf = b''

    def expand(self, n):
        while len(self.buf) < n:
            self.buf += self.receiver.recv()
        return self.buf

    def shift(self, n=1):
        if self.buf:
            #logger.info('drop bytes, number of bytes:' + str(n))
            self.buf = self.buf[n:]


class NdasReceiver:

    def __init__(self, conn_str):
        self.buffer_manager = ReceiverBufferManager(conn_str)

    def recv(self, with_bin_data=False):
        while True:
            self.buffer_manager.expand(5)
            if self.buffer_manager.buf[4:5] != b'{':
                logger.warning('no json start symbol:' + str(self.buffer_manager.buf[:5]))
                self.buffer_manager.shift()
                continue
            size_bytes = self.buffer_manager.buf[:4]
            size = int.from_bytes(size_bytes, byteorder='little')
            if not 20 < size < 20000:
                logger.error('possibly incorrect size:' + str(size))
                self.buffer_manager.shift()
                continue
            self.buffer_manager.expand(size + 4)
            last_byte_bstr = self.buffer_manager.buf[size + 3:size + 4]
            if last_byte_bstr != b'}':
                logger.error('incorrect last byte:' + str(last_byte_bstr))
                self.buffer_manager.shift()
                continue
            bdata = self.buffer_manager.buf[4:size + 4]
            try:
                json_data = json.loads(bdata.decode('utf-8'))
            except Exception as e:
                logger.error('cannot parse json data:\n' + str(bdata) + '\n' + str(e))
                self.buffer_manager.shift()
                continue
            if with_bin_data:
                bdata = self.buffer_manager.buf[:size + 4]
                buf_item = {'bin': bdata}
                if 'signal' in json_data:
                    buf_item['timestmp'] = json_data['signal']['timestmp']
                ret_val = json_data, buf_item
            else:
                ret_val = json_data
            self.buffer_manager.shift(size + 4)
            return ret_val

