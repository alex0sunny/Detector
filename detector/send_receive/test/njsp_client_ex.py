# Client example
from detector.send_receive.njsp.njsp import NJSP_MULTISTREAMREADER

streamreader = NJSP_MULTISTREAMREADER()
njsp_client_params = {
    'subscriptions': ['streams', 'parameters', 'status'],
    'flush_buffer': False,
    'client_name': 'trigger'
}
streamreader.add_client('192.168.0.240', 10000, params=njsp_client_params)
# streamreader.add_client('192.168.0.240', 10001, params=njsp_client_params)
# streamreader.add_client('192.168.0.240', 10002, params=njsp_client_params)

while True:
    try:
        packet = streamreader.queue.get(timeout=1)
    except:
        packet = None
    if packet:
        print(packet)
#streamreader.connected_event.wait()
# stream_name = list(streamreader.init_packet['parameters']['streams'].keys())[0]
# channel_name = list(streamreader.init_packet['parameters']['streams'][stream_name]['channels'].keys())[0]
# while streamreader.connected_event.is_set():
#     try:
#         packet = streamreader.queue.get(timeout=0.5)
#     except Exception:
#         packet = None
#     if packet != None and ('streams' in packet):
#         data_bytes = packet['streams'][stream_name]['samples'][channel_name]
#         val = int.from_bytes(data_bytes, byteorder='little', signed=True)
#         print(val)
# streamreader.kill()
