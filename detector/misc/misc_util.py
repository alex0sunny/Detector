from time import sleep

import zmq
from obspy import UTCDateTime

from detector.misc.globals import Port, Subscription


def get_expr(formula_list, triggers_dic):
    expr_list = []
    for i in range(len(formula_list)):
        expr_item = formula_list[i]
        if i % 2:
            expr_list.append(expr_item)
        else:
            expr_list.append(str(triggers_dic.get(int(expr_item), False)))
    return ' '.join(expr_list)


def get_formula_triggers(formula_list):
    triggers = []
    for i in range(len(formula_list)):
        if not i % 2:
            trigger_id = int(formula_list[i])
            triggers.append(trigger_id)
    return triggers


def to_action_rules(rule_actions):
    action_rules = {}
    for rule, actions in rule_actions.items():
        for action in actions:
            if action not in action_rules:
                action_rules[action] = []
            action_rules[action].append(rule)
    return action_rules


# def get_channels(context, stations_set):
#     socket_sub = context.socket(zmq.SUB)
#     socket_sub.connect('tcp://localhost:' + str(Port.proxy.value))
#     socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.intern.value)
#
#     local_set = {}
#     chs_set = {}
#
#     cur_time = UTCDateTime()
#     check_time = cur_time + 2
#     while cur_time < check_time:
#         sleep(.1)
#         try:
#             bdata = socket_sub.recv(zmq.NOBLOCK)
#         except zmq.ZMQError:
#             pass
#
#     socket_sub.close()

#print(get_formula_triggers(['1', 'or', '2', 'and', '3']))
#print(get_expr(['2', 'and not', '3', 'and', '1'], {1: True, 2: True}))

