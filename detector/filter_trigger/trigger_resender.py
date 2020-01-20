import zmq
from obspy import *

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port
from detector.send_receive.tcp_server import TcpServer


def resend(conn_str, channels, pem, pet):
    context = zmq.Context()

    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:%d' % Port.internal_resend.value)
    socket_sub.setsockopt(zmq.SUBSCRIBE, b'')

    socket_server = TcpServer(conn_str, context)

    event_conn_str = 'tcp://*:%d' % Port.trigger.value

    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.bind(event_conn_str)
    for channel in channels:
        socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01' + channel.encode())

    test_send = False
    trigger = False
    buf = []
    pet_time = UTCDateTime(0)
    while True:
        try:
            bin_data = socket_trigger.recv(zmq.NOBLOCK)
            trigger_data = bin_data[-1:]
            trigger_time = UTCDateTime(int.from_bytes(bin_data[-9:-1], byteorder='big') / 10**9)
            if trigger and trigger_data[-1:] == b'0':
                trigger = False
                pet_time = trigger_time + pet
                logger.info('detriggered\ndetrigger time:' + str(trigger_time) + '\npet time:' +
                            str(trigger_time + pet) + '\nchannel:' + str(bin_data[4:-9]))
            if not trigger and trigger_data[-1:] == b'1':
                trigger = True
                logger.info('triggered\ntrigger time:' + str(trigger_time) + '\npem time:' +
                            str(trigger_time - pem) + '\nchannel:' + str(bin_data[4:-9]))
                if buf:
                    logger.info('buf item dt:' + str(buf[0][0]))
            if not buf:
                logger.warning('buf is empty')
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
                if not test_send:
                    socket_server.send(buf[0][1])
                # logger.debug('buf item dt:' + str(buf[0][0]))
                buf = buf[1:]
            # logger.debug('send regular data, dt' + str(dt))
            socket_server.send(bdata)
        else:
            if test_send and buf:
                socket_server.send(buf[0][1])
            #logger.debug('append to buf with dt:' + str(dt))
            buf.append((dt, bdata))
        #logger.debug('buf[0]:' + str(buf[0]) + '\nbuf[0][0]:' + str(buf[0][0]))
        if buf:
            dt_begin = buf[0][0]
            while dt_begin < dt - pem and buf[3:]:
                # logger.debug('delete from buf with dt:' + str(buf[0][0]) + '\ncurrent pem:' +
                #              str(dt-pem) + '\ncurrent buf:' + str(buf[0][0]) + '-' + str(buf[-1][0]))
                buf = buf[1:]
                dt_begin = buf[0][0]
