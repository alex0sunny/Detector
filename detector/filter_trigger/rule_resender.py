from _ctypes import sizeof
from ctypes import memmove, addressof
from io import BytesIO

import zmq
from obspy import *

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, Subscription
from detector.misc.header_util import CustomHeader
from detector.send_receive.njsp_server import NjspServer
from detector.send_receive.tcp_server import TcpServer


def resend(conn_str, rules, pem, pet):
    logger.debug('start rule resender')
    context = zmq.Context()

    socket_sub = context.socket(zmq.SUB)
    conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)
    socket_sub.connect(conn_str_sub)
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.parameters.value)
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.signal.value)

    socket_confirm = context.socket(zmq.PUB)
    socket_confirm.connect('tcp://localhost:' + str(Port.multi.value))
    
    socket_server = NjspServer(conn_str, context)

    socket_rule = context.socket(zmq.SUB)
    socket_rule.connect(conn_str_sub)
    socket_rule.setsockopt(zmq.SUBSCRIBE, Subscription.test.value + b'03')
    for rule_index in rules:
        rule_index_s = '%02d' % rule_index
        socket_rule.setsockopt(zmq.SUBSCRIBE, Subscription.rule.value + rule_index_s.encode())

    trigger = False
    buf = []
    pet_time = UTCDateTime(0)
    while True:
        try:
            bin_data = socket_rule.recv(zmq.NOBLOCK)
            test = bin_data[:1] == Subscription.test.value
            if test:
                logger.debug('test rule event')
                trigger_data = b'0'
                if buf:
                    trigger_time, _ = buf[-1]
                else:
                    trigger_time = None
            else:
                logger.debug('rule event')
                trigger_data = bin_data[3:4]
                trigger_time = UTCDateTime(int.from_bytes(bin_data[-8:], byteorder='big') / 10**9)
            if trigger and trigger_data == b'0' or test:
                trigger = False
                if test and not trigger_time:
                    pet_time = None
                    logger.info('detrigger test event')
                else:
                    pet_time = trigger_time + pet
                    logger.info('detriggered\ndetrigger time:' + str(trigger_time) + '\npet time:' +
                                str(trigger_time + pet) + '\ntrigger:' + str(bin_data[1:3]))
            if not trigger and trigger_data == b'1':
                trigger = True
                logger.info('triggered\ntrigger time:' + str(trigger_time) + '\npem time:' +
                            str(trigger_time - pem) + '\ntrigger:' + str(bin_data[1:3]))
                if buf:
                    logger.info('buf item dt:' + str(buf[0][0]))
            if not buf:
                logger.warning('buf is empty')
        except zmq.ZMQError:
            pass

        socket_confirm.send(Subscription.confirm.value + b'1')
        if not socket_sub.poll(3000):
            logger.info('no signal or params data')
            continue
        raw_data = socket_sub.recv()
        #print('raw_data recvd:' + str(raw_data))
        if raw_data[:1] == Subscription.parameters.value:
            logger.debug('parameters received in resender')
            #exit(1)
            parameters_bytes = raw_data[1:]
            socket_server.set_params(parameters_bytes)
            continue
        resent_data = raw_data[1:]
        custom_header = CustomHeader()
        header_size = sizeof(CustomHeader)
        BytesIO(resent_data[:header_size]).readinto(custom_header)
        #memmove(addressof(custom_header), resent_data[:header_size], header_size)
        # logger.debug('custom header received:' + str(custom_header))
        dt = UTCDateTime(custom_header.ns / 10 ** 9)
        #logger.debug('dt:' + str(dt))
        # logger.debug('wait binary data')
        bdata = resent_data[header_size:]
        # logger.debug('binary data received')
        #logger.debug('dt:' + str(UTCDateTime(dt)) + ' bdata len:' + str(len(bdata)))
        if not pet_time:
            #logger.debug('pet time is None')
            pet_time = dt + pet
        if dt < pet_time or trigger:
            #logger.debug('dt:' + str(dt) + ', pet time:' + str(pet_time) + ', trigger:' + str(trigger))
            # if buf:
            #     logger.debug('clear buf, trigger:' + str(trigger))
            while buf:
                #logger.debug('send data to output from buf')
                socket_server.send(buf[0][1])
                # logger.debug('buf item dt:' + str(buf[0][0]))
                buf = buf[1:]
            # logger.debug('send regular data, dt' + str(dt))
            #logger.debug('send data to output')
            socket_server.send(bdata)
        else:
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
