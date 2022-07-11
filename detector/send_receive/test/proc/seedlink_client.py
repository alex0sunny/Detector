from threading import Thread
from time import sleep

from obspy import *
from obspy.clients.seedlink.easyseedlink import *
from matplotlib import pyplot

from obspy.clients.seedlink.basic_client import Client


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


#client_us = create_client('rtserve.iris.washington.edu', on_data=handle_data)
#client_fr = create_client('rtserver.ipgp.fr', on_data=handle_data)
#client = create_client('192.168.0.178', on_data=handle_data)
client = create_client('192.168.0.240', on_data=handle_data)
#client = create_client('81.26.81.45', on_data=handle_data)
print(client.get_info('streams'))
#exit(1)
print()
print(client.get_info('ALL'))
#exit(1)
print()
#print(client_fr.get_info('streams'))
#print()
#print(client_fr.get_info('ALL'))


#client_us.select_stream('IU', 'ANMO', 'BH?')
#client_fr.select_stream('RU', 'DEVA', 'CH?')
#client.select_stream('KS', 'SHAR', 'S2?')
#client.select_stream('RU', 'ND01', 'CH?')
client.select_stream('RU', 'ND02', 'CH?')
#client.select_stream('RU', 'ND14', 'C??')
#client.select_stream('RU', 'NADC', 'CH?')


#geofon.gfz-potsdam.de


# def f_receiver_fr():
#     client_fr.run()


def f_receiver():
    client.run()


# Thread(target=f_receiver_fr).start()
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

