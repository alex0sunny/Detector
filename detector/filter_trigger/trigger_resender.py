import zmq
from obspy import *

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.server_zmq import ZmqServer


def resend(conn_str, channel):
    context = zmq.Context()

    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:5560')
    socket_sub.setsockopt(zmq.SUBSCRIBE, b'')

    socket_server = ZmqServer(conn_str, context)

    event_conn_str = 'tcp://*:5562'

    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.bind(event_conn_str)
    socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01' + channel.encode())

    trigger = False
    while True:
        try:
            trigger_data = socket_trigger.recv(zmq.NOBLOCK)[-1:]
            if trigger and trigger_data[-1:] == b'0':
                trigger = False
                logger.info('detriggered')
            if not trigger and trigger_data[-1:] == b'1':
                trigger = True
                logger.info('triggered')
        except zmq.ZMQError:
            pass
        dt_bytes = socket_sub.recv()
        dt = UTCDateTime(int.from_bytes(dt_bytes, byteorder='big') / 10 ** 9)
        bdata = socket_sub.recv()
        #logger.debug('dt:' + str(UTCDateTime(dt)) + ' bdata len:' + str(len(bdata)))
        if trigger:
            socket_server.send(bdata)
