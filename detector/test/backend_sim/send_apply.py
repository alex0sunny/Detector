import zmq
import time

from detector.misc.globals import Port

context = zmq.Context()
socket_backend = context.socket(zmq.PUB)
socket_backend.connect('tcp://localhost:%d' % Port.backend.value)
socket_backend.send(b'AP')
time.sleep(.1)
socket_backend.send(b'AP')

