import zmq

from detector.misc.globals import Port, logger


def triggers_proxy():

    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.bind('tcp://*:' + str(Port.trigger.value))
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:' + str(Port.proxy.value))

    while True:
        mes = socket_sub.recv()
        socket_pub.send(mes)

