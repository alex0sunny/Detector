import zmq
from obspy import *

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.server_zmq import ZmqServer


def resend(conn_str, channels, pem, pet):
    context = zmq.Context()

    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:5560')
    socket_sub.setsockopt(zmq.SUBSCRIBE, b'')

    socket_server = ZmqServer(conn_str, context)

    event_conn_str = 'tcp://*:5562'

    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.bind(event_conn_str)
    for channel in channels:
        socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01' + channel.encode())

    trigger = False
    buf = []
    pet_time = UTCDateTime(0)
    while True:
        try:
            trigger_data = socket_trigger.recv(zmq.NOBLOCK)[-1:]
            if trigger and trigger_data[-1:] == b'0':
                trigger = False
                pet_time = dt + pet
                logger.info('detriggered\ndetrigger time:' + str(dt) + '\npet time:' + str(dt + pet))
            if not trigger and trigger_data[-1:] == b'1':
                trigger = True
                logger.info('triggered\ntrigger time:' + str(dt) + '\npem time:' + str(dt - pem) +
                            '\ncurrent buf:' + str(buf[0][0]) + '-' + str(buf[-1][0]))
        except zmq.ZMQError:
            pass
        dt_bytes = socket_sub.recv()
        dt = UTCDateTime(int.from_bytes(dt_bytes, byteorder='big') / 10 ** 9)
        bdata = socket_sub.recv()
        #logger.debug('dt:' + str(UTCDateTime(dt)) + ' bdata len:' + str(len(bdata)))
        if dt < pet_time or trigger:
            # if buf:
            #     logger.debug('clear buf, trigger:' + str(trigger))
            while buf:
                socket_server.send(buf[0][1])
                # logger.debug('buf item dt:' + str(buf[0][0]))
                buf = buf[1:]
            # logger.debug('send regular data, dt' + str(dt))
            socket_server.send(bdata)
        else:
            buf.append((dt, bdata))
        #logger.debug('buf[0]:' + str(buf[0]) + '\nbuf[0][0]:' + str(buf[0][0]))
        if buf:
            dt_begin = buf[0][0]
            while dt_begin < dt - pem:
                buf = buf[1:]
                dt_begin = buf[0][0]
