import time
import zmq

from obspy import UTCDateTime

from detector.filter_trigger.StaLtaTrigger import logger
from detector.send_receive.client_zmq import ZmqClient

context = zmq.Context()
client = ZmqClient('tcp://192.168.0.189:5555', context)

while True:
    data = client.recv(13)
    logger.debug('received:' + str(data))

