import zmq

from detector.misc.globals import Port, Subscription


def action_process(action_id, send_func, rules=[], args={}):
    if action_id in [1, 2]:
        print('start process for ' + str(action_id))
    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:' + str(Port.proxy.value))
    action_id_s = '%02d' % action_id
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.test.value + action_id_s.encode())
    for rule_id in rules:
        rule_id_s = '%02d' % rule_id
        socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.rule.value + rule_id_s.encode())
    while True:
        bin_message = socket_sub.recv()
        if action_id in [1, 2]:
            args['relay_n'] = action_id
            print('message received for action ' + str(action_id) + ', mes:' + str(bin_message))
        args['on_off'] = int(bin_message[3:4])
        send_func(**args)

