# Client example
from detector.send_receive.njsp.njsp import NJSP_STREAMREADER

streamreader = NJSP_STREAMREADER(('localhost', 12345))
streamreader.connected_event.wait()
stream_name = list(streamreader.init_packet['parameters']['streams'].keys())[0]
channel_name = list(streamreader.init_packet['parameters']['streams'][stream_name]['channels'].keys())[0]
while streamreader.connected_event.is_set():
    try: packet = streamreader.queue.get(timeout=0.5)
    except: packet = None
    if packet != None and ('streams' in packet):
        data_bytes = packet['streams'][stream_name]['samples'][channel_name]
        val = int.from_bytes(data_bytes,byteorder='little',signed=True)
        print(val)
streamreader.kill()


