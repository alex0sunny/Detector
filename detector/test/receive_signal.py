import zmq

from obspy import *
from matplotlib import pyplot

from detector.misc.header_util import bin_to_stream


def receive_signal():

    context = zmq.Context.instance()
    socket = context.socket(zmq.STREAM)

    socket.connect('tcp://localhost:5555')
    socket.connect('tcp://localhost:5565')
    # for [host, port] in conn_tuples:
    #     socket.connect('tcp://' + host + ':' + str(port))

    st = Stream()

    pyplot.ion()
    figure = pyplot.figure()

    while True:
        id = socket.expand()
        bin_data = socket.expand()
        if not bin_data:
            print('empty data')
        else:
            #print('bin_data size:' + str(len(bin_data)))
            st += bin_to_stream(bin_data)
            st.merge()
            print('current stream:' + str(st))
            #print("received:" + str(st))
            if (st[0].stats.endtime - st[0].stats.starttime) > 6:
                print('show stream..')
                endtime = max([tr.stats.endtime for tr in st])
                st = st.slice(endtime - 5)
                pyplot.clf()
                st.plot(fig=figure)
                pyplot.show()


receive_signal()



