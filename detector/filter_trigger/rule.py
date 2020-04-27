from time import sleep

import zmq
from obspy import UTCDateTime

from detector.misc.globals import Port, Subscription, logger
from detector.misc.misc_util import get_formula_triggers, get_expr


def rule_picker(rule_id, formula_list):
    triggers = get_formula_triggers(formula_list)
    rule_on = False
    rule_id_s = '%02d' % rule_id
    triggers_dic = {trigger_id: False for trigger_id in triggers}
    context = zmq.Context()
    socket_rule = context.socket(zmq.PUB)
    socket_rule.connect('tcp://localhost:' + str(Port.multi.value))
    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.connect('tcp://localhost:' + str(Port.proxy.value))
    for trigger_id in triggers:
        trigger_id_s = '%02d' % trigger_id
        socket_trigger.setsockopt(zmq.SUBSCRIBE, Subscription.trigger.value + trigger_id_s.encode())
    trigger_buf = []
    while True:
        try:
            bdata = socket_trigger.recv(zmq.NOBLOCK)[1:]
            trigger_id_s = bdata[:2].decode()
            trigger_id = int(trigger_id_s)
            trigger_val = bdata[2:3] == b'1'
            trigger_time = UTCDateTime(int.from_bytes(bdata[-8:], byteorder='big') / 10**9)
            # logger.debug('rule id:' + rule_id_s + ' trigger_id:' + trigger_id_s + ' trigger time:' +
            #              str(trigger_time))
            trigger_buf.append((trigger_time, trigger_id, trigger_val))
        except zmq.ZMQError:
            sleep(.1)
            check_time = UTCDateTime() - 1.5
            trigger_times = sorted([t for t, _, _ in trigger_buf if t < check_time])
            if trigger_times:
                trigger_time = trigger_times[0]
                #logger.debug('process trigger, trigger time:' + str(trigger_time))
                for i in range(len(trigger_buf)):
                    buf_item = trigger_buf[i]
                    t, trigger_id, trigger_val = buf_item
                    if t == trigger_time:
                        #logger.debug('trigger id:' + str(trigger_time))
                        trigger_buf.pop(i)
                        break
                triggers_dic[trigger_id] = trigger_val
                #logger.debug('trigger dic:' + str(triggers_dic))
                rule_expr = get_expr(formula_list, triggers_dic)
                #logger.debug('rule expr:' + rule_expr)
                rule_val = eval(rule_expr)
                if rule_val != rule_on:
                    logger.info('rule triggered, rule_id:' + rule_id_s + ' rule val:' + str(rule_val) +
                                ' rule expr:' + rule_expr)
                    rule_on = rule_val
                    bin_message = Subscription.rule.value + rule_id_s.encode()
                    if rule_on:
                        bin_message += b'1'
                    else:
                        bin_message += b'0'
                    bin_message += trigger_time._ns.to_bytes(8, byteorder='big')
                    logger.debug('bin_message:' + str(bin_message))
                    socket_rule.send(bin_message)

