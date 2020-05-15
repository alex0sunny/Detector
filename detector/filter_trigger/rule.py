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
    while True:
        bdata = socket_trigger.recv()[1:]
        trigger_id_s = bdata[:2].decode()
        trigger_id = int(trigger_id_s)
        trigger_val = bdata[2:3] == b'1'
        triggers_dic[trigger_id] = trigger_val
        rule_expr = get_expr(formula_list, triggers_dic)
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
            bin_message += bdata[-8:]
            logger.debug('bin_message:' + str(bin_message))
            socket_rule.send(bin_message)

