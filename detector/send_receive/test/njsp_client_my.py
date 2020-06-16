# Client example
import numpy as np
from detector.send_receive.njsp.njsp import NJSP_STREAMREADER

streamreader = NJSP_STREAMREADER(('localhost', 10001))
streamreader.connected_event.wait()
print('connected')
stream_name = list(streamreader.init_packet['parameters']['streams'].keys())[0]
channel_name = list(streamreader.init_packet['parameters']['streams'][stream_name]['channels'].keys())[0]
while streamreader.connected_event.is_set():
    try: packet = streamreader.queue.get(timeout=0.5)
    except: packet = None
    if packet != None and ('streams' in packet):
        data_bytes = packet['streams'][stream_name]['samples'][channel_name]
        #val = int.from_bytes(data_bytes,byteorder='little',signed=True)
        bin_signal = np.frombuffer(data_bytes, dtype='int32')
        print(bin_signal)
streamreader.kill()


