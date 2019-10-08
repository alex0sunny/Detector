import pickle

import zmq
from matplotlib import pyplot

pyplot.ion()
figure = pyplot.figure()

context = zmq.Context()
socket = context.socket(zmq.SUB)

socket.connect("tcp://127.0.0.1:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, '')
print('socket connected')
while True:
    st = pickle.loads(socket.recv())
    print('signal received')
    pyplot.clf()
    st.plot(fig=figure)
    pyplot.show()
    pyplot.pause(.01)

