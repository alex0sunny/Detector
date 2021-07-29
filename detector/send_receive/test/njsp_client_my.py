# Client example
import numpy as np
from detector.send_receive.njsp.njsp import NJSP_STREAMREADER

streamreader = NJSP_STREAMREADER(('192.168.0.189', 10010))

streamreader.connected_event.wait()
print('connected\ninit_packet:\n', streamreader.init_packet)
stream_name = list(streamreader.init_packet['parameters']['streams'].keys())[0]
channel_name = list(streamreader.init_packet['parameters']['streams'][stream_name]['channels'].keys())[0]
while streamreader.connected_event.is_set():
    try: packet = streamreader.queue.get(timeout=0.5)
    except: packet = None
    if packet != None and ('streams' in packet):
        print('streams:', str(list(packet['streams'].keys())))
        #data_bytes = packet['streams'][stream_name]['samples'][channel_name]
        #print('type:', type(data_bytes), 'decoded:', data_bytes.decode('utf16'))
        #val = int.from_bytes(data_bytes,byteorder='little',signed=True)
        #bin_signal = np.frombuffer(data_bytes, dtype='int32')
        print('received data packet')
    if packet != None and ('streams' not in packet):
        print('parameters:', packet)

streamreader.kill()


