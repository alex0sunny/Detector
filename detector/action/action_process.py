import zmq

from detector.action.relay_actions import turn
from detector.misc.globals import Port, Subscription


def action_process(action_id, send_func, args={}):
    relay_action = action_id in [1, 2]
    if relay_action:
        args = {'relay_n': action_id}
    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:' + str(Port.proxy.value))
    action_id_s = '%02d' % action_id
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.test.value + action_id_s.encode())
    while True:
        bin_message = socket_sub.recv()
        if relay_action:
            args['on_off'] = int(bin_message[-1:])
        send_func(**args)

