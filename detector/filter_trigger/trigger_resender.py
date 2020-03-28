from _ctypes import sizeof
from ctypes import memmove, addressof
from io import BytesIO

import zmq
from obspy import *

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, sources_dic, Subscription
from detector.misc.header_util import CustomHeader
from detector.send_receive.tcp_server import TcpServer


def resend(conn_str, triggers, pem, pet):
    context = zmq.Context()

    socket_sub = context.socket(zmq.SUB)
    conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)
    socket_sub.connect(conn_str_sub)
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.signal.value)

    socket_server = TcpServer(conn_str, context)

    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.connect(conn_str_sub)
    for trigger_index in triggers:
        trigger_index_s = '%02d' % trigger_index
        socket_trigger.setsockopt(zmq.SUBSCRIBE, Subscription.trigger.value + trigger_index_s.encode())

    test_send = False
    trigger = False
    buf = []
    pet_time = UTCDateTime(0)
    while True:
        try:
            bin_data = socket_trigger.recv(zmq.NOBLOCK)[1:]
            logger.debug('trigger event')
            trigger_data = bin_data[2:3]
            trigger_time = UTCDateTime(int.from_bytes(bin_data[-8:], byteorder='big') / 10**9)
            if trigger and trigger_data == b'0':
                trigger = False
                pet_time = trigger_time + pet
                logger.info('detriggered\ndetrigger time:' + str(trigger_time) + '\npet time:' +
                            str(trigger_time + pet) + '\ntrigger:' + str(bin_data[:2]))
            if not trigger and trigger_data == b'1':
                trigger = True
                logger.info('triggered\ntrigger time:' + str(trigger_time) + '\npem time:' +
                            str(trigger_time - pem) + '\ntrigger:' + str(bin_data[:2]))
                if buf:
                    logger.info('buf item dt:' + str(buf[0][0]))
            if not buf:
                logger.warning('buf is empty')
        except zmq.ZMQError:
            pass

        # logger.debug('wait custom header')
        resent_data = socket_sub.recv()[1:]
        custom_header = CustomHeader()
        header_size = sizeof(CustomHeader)
        BytesIO(resent_data[:header_size]).readinto(custom_header)
        #memmove(addressof(custom_header), resent_data[:header_size], header_size)
        # logger.debug('custom header received:' + str(custom_header))
        dt = UTCDateTime(custom_header.ns / 10 ** 9)
        # logger.debug('wait binary data')
        bdata = resent_data[header_size:]
        # logger.debug('binary data received')
        #logger.debug('dt:' + str(UTCDateTime(dt)) + ' bdata len:' + str(len(bdata)))
        if dt < pet_time or trigger:
            # if buf:
            #     logger.debug('clear buf, trigger:' + str(trigger))
            while buf:
                if test_send:
                    socket_server.send(buf[0][1])
                # logger.debug('buf item dt:' + str(buf[0][0]))
                buf = buf[1:]
            # logger.debug('send regular data, dt' + str(dt))
            if test_send:
                socket_server.send(bdata)
        else:
            if test_send and buf:
                socket_server.send(buf[0][1])
            # logger.debug('append to buf with dt:' + str(dt))
            buf.append((dt, bdata))
        if buf:
            # logger.debug('buf[0]:' + str(buf[0]) + '\nbuf[0][0]:' + str(buf[0][0]))
            dt_begin = buf[0][0]
            while dt_begin < dt - pem and buf[3:]:
                # logger.debug('delete from buf with dt:' + str(buf[0][0]) + '\ncurrent pem:' +
                #              str(dt-pem) + '\ncurrent buf:' + str(buf[0][0]) + '-' + str(buf[-1][0]))
                buf = buf[1:]
                dt_begin = buf[0][0]
        # else:
        #     logger.debug('buf is empty')
