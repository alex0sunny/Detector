import time

import zmq
from matplotlib import pyplot
from obspy import *
from multiprocessing import Process
import socket

#from detector.test import send_signal, receive_signal
from detector.header_util import chunk_stream, stream_to_bin, bin_to_stream
from detector.test.signal_generator import SignalGenerator


def send_signal(st, host, port):

    signal_generator = SignalGenerator(st)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(10)
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                st = signal_generator.get_stream()
                sts = chunk_stream(st)
                bin_datas = [stream_to_bin(st) for st in sts]
                for bin_data in bin_datas:
                    print('bdata size:' + str(len(bin_data)))
                    conn.sendall(bin_data)
                    time.sleep(.01)
                time.sleep(.5)


def receive_signal(conn_tuples):

    context = zmq.Context.instance()
    socket = context.socket(zmq.STREAM)

    # socket.connect('tcp://localhost:5555')
    # socket.connect('tcp://localhost:5565')
    for [host, port] in conn_tuples:
        socket.connect('tcp://' + host + ':' + str(port))

    st = Stream()

    pyplot.ion()
    figure = pyplot.figure()

    while True:
        id = socket.recv()
        bin_data = socket.recv()
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


def bandpass_filter(st, conn_str_sub, conn_str_pub):
    context = zmq.Context.instance()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:5566')
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:5567')
    while True:
        bin_data = socket_sub.recv()


st = read('D:/converter_data/example/onem.mseed')
for tr in st:
    tr.stats.station = 'ND01'

st2 = read('D:/converter_data/example/threeCh100.mseed')
st2 = Stream() + st2[0] + st2[-1]
for tr in st2:
    tr.stats.station = 'ND02'


if __name__ == '__main__':
    p_sender1000 = Process(target=send_signal, args=(st, 'localhost', 5555))
    p_sender200 = Process(target=send_signal, args=(st2, 'localhost', 5565))
    #arg_receiver = [('localhost', 5555), ('localhost', 5565)]
    p_receiver = Process(target=receive_signal, args=([('localhost', 5555), ('localhost', 5565)],))

    p_sender1000.start()
    p_sender200.start()
    p_receiver.start()

    time.sleep(20)
    p_sender1000.terminate()
    p_sender200.terminate()
    p_receiver.terminate()


