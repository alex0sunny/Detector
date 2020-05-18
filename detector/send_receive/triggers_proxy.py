import zmq

from detector.misc.globals import Port, logger, Subscription


def triggers_proxy():

    trigger_delay = 2000

    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.bind('tcp://*:' + str(Port.multi.value))
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, '')
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:' + str(Port.proxy.value))

    while True:
        if not socket_sub.poll(100):
            continue
        mes = socket_sub.recv()
        if mes[0] != Subscription.trigger.value:
            socket_pub.send(mes)
            continue
        # implement trigger delay
        socket_pub.send(mes)

