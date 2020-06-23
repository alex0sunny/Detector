# Server example
import time

from detector.send_receive.njsp.njsp import NJSP_STREAMSERVER

init_packet = {
    'parameters': {
        'streams': {
            'main': {
                'channels': {
                    'ch1': {
                        'ch_active': True
                    }
                }
            }
        }
    }
}

streamserver = NJSP_STREAMSERVER(('localhost', 12345), init_packet)
counter = 0
while counter <= 100:
    data_packet = {
        'streams': {
            'main': {
                'samples': {
                    'ch1': counter.to_bytes(4, byteorder='little', signed=True)
                }
            }
        }
    }
    streamserver.broadcast_data(data_packet)
    print(counter)
    counter += 1
    time.sleep(1)
streamserver.kill()

