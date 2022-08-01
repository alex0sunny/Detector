import time

from obspy import UTCDateTime


def get_expr(formula_list, triggers_dic):
    expr_list = []
    for i in range(len(formula_list)):
        expr_item = formula_list[i]
        if i % 2:
            expr_list.append(expr_item)
        else:
            val = triggers_dic[expr_item]
            expr_list.append(str(val))
    return ' '.join(expr_list)


def group_triggerings(triggerings, user_triggerings, last_triggerings={}):
    for date_time, triggering, trigger_id in triggerings:
        last_triggerings[trigger_id] = triggering
        user_triggerings[trigger_id].append(triggering)
        user_triggerings[trigger_id][:-10] = []


def fill_out_triggerings(triggers_ids, user_triggerings, last_triggerings):
    triggerings_out = {}
    for trigger_id in triggers_ids:
        if user_triggerings[trigger_id]:
            triggerings_out[trigger_id] = user_triggerings[trigger_id].pop(0)
        else:
            triggerings_out[trigger_id] = last_triggerings[trigger_id]
    return triggerings_out


def to_actions_triggerings(rules_triggerings, rules_settings, actions_triggerings):
    for date_time, triggering, rule_id in rules_triggerings:
        for action_id in rules_settings[rule_id]['actions']:
            actions_triggerings.append((date_time, 1 if triggering else -1, action_id))


def append_test_triggerings(actions_triggerings, test_triggerings):
    for action_id in test_triggerings:
        triggering = test_triggerings[action_id]
        actions_triggerings.append((float(UTCDateTime()), triggering, action_id))
        if triggering == 1:
            test_triggerings[action_id] = -1
        if triggering == -1:
            test_triggerings[action_id] = 0


def to_action_rules(rule_actions):
    action_rules = {}
    for rule, actions in rule_actions.items():
        for action in actions:
            if action not in action_rules:
                action_rules[action] = []
            action_rules[action].append(rule)
    return action_rules


def f_empty():
    while True:
        time.sleep(2)


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

