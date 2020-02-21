from obspy import *
from obspy.clients.seedlink.easyseedlink import *
from matplotlib import pyplot


pyplot.ion()
figure = pyplot.figure()
st = Stream()
check_time = UTCDateTime()


def handle_data(trace):
    print('Received the following trace:')
    print(trace)
    print()
    global st, check_time
    st += trace
    cur_time = UTCDateTime()
    if cur_time > check_time + 1:
        check_time = cur_time
        st.sort().merge()
        st.trim(starttime=st[-1].stats.endtime - 60)
        pyplot.clf()
        st.plot(fig=figure)
        pyplot.show()
        pyplot.pause(.1)


#client = create_client('rtserve.iris.washington.edu', on_data=handle_data)
#client = create_client('192.168.0.200', on_data=handle_data)
client = create_client('81.26.81.45', on_data=handle_data)
print(client.get_info('streams'))
print()
print(client.get_info('ALL'))

#client.select_stream('IU', 'ANMO', 'BH?')
#client.select_stream('KS', 'SHAR', 'S2?')
client.select_stream('KS', 'SHAZ', 'SH?')

client.run()

