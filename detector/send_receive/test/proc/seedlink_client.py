from threading import Thread
from time import sleep

from obspy import *
from obspy.clients.seedlink.easyseedlink import *
from matplotlib import pyplot


pyplot.ion()
figure = pyplot.figure()
st = Stream()


def handle_data(trace):
    print('trace received:' + str(trace))
    global st
    st += trace
    endtime = st[-1].stats.endtime
    if endtime > st[0].stats.starttime + 70:
        st.trim(endtime - 60)
    #print('trace %s appended to stream %s' % (trace, st))


#client = create_client('rtserve.iris.washington.edu', on_data=handle_data)
client = create_client('192.168.0.226', on_data=handle_data)
#client = create_client('81.26.81.45', on_data=handle_data)
print(client.get_info('streams'))
print()
print(client.get_info('ALL'))

#client.select_stream('IU', 'ANMO', 'BH?')
#client.select_stream('KS', 'SHAR', 'S2?')
client.select_stream('RU', 'ND01', 'DN?')


def f_receiver():
    client.run()


Thread(target=f_receiver).start()


while True:
    if not st:
        sleep(1)
        continue
    st_vis = st[:]
    st_vis.sort().merge()
    endtime = st_vis[-1].stats.endtime
    st_vis.trim(endtime - 60)
    #print('plot stream %s' % st)
    pyplot.clf()
    st_vis.plot(fig=figure)
    pyplot.show()
    pyplot.pause(1)

