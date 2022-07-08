from obspy import UTCDateTime

import detector.misc.globals as glob
from detector.action.action_pipe import execute_action
from detector.misc.globals import ActionType
from copy import deepcopy
from queue import Queue, Empty
from time import sleep

from backend.trigger_html_util import getSources, get_actions
from detector.action.action_process import main_action
from detector.misc.globals import logger

base_params = {
    'reconnect': True,
    'reconnect_period': 30,
    'bson': True,
    'handshake': {
        'subscriptions': ['status', 'log', 'streams', 'alarms'],
        'flush_buffer': False,
        'client_name': 'NOTSET'
    }
}


def worker(njsp):

    while True:

        glob.restart = False
        njsp_queue = Queue(100)
        packets_q = []
        readers = {}
        streamers = {}
        sample_rates = {}

        counters = {}
        pet_times = {}

        sources = getSources()
        for station in sources:
            njsp_params = deepcopy(base_params)
            njsp_params['handshake']['client_name'] = station
            conn_data = sources[station]
            readers[station] = njsp.add_reader(conn_data['host'], int(conn_data['port']), station,
                                               njsp_params, njsp_queue)

        glob.TEST_TRIGGERINGS = {}
        actions_settings = get_actions()
        for action_id in [ActionType.relay_A.value, ActionType.relay_B.value]:
            execute_action(action_id, 0, actions_settings[action_id].get('inverse', False))
        for action_id in actions_settings:
            counters[action_id] = pet_times[action_id] = 0
            glob.TEST_TRIGGERINGS[action_id] = 0
            # glob.TEST_TRIGGERINGS[action_id] = -1 if action_id in \
            #     [ActionType.relay_A.value, ActionType.relay_B.value] else 0

        while not glob.restart:
            try:
                packets_data = njsp_queue.get(timeout=1)
                for conn_name, dev_packets in packets_data.items():
                    station = conn_name.split(':')[1]
                    for packet_type, content in dev_packets.items():
                        packet_type, content = rename_packet(packet_type, content, station)
                        if 'parameters' == packet_type and station not in streamers:
                            streamer_params = {'init_packet': {'parameters': content.copy()},
                                               'ringbuffer_size': 10}
                            streamers[station] = njsp.add_streamer('', sources[station]['out_port'],
                                                                   streamer_params)
                            sample_rates[station] = content['streams'][station]['sample_rate']
                        if 'streams' == packet_type:
                            packets_q.append({packet_type: content})
            except Empty:
                #logger.info('no data')
                pass
                # logger.debug('packets:\n' + str(packets_data))
            packets_q[:-glob.PBUF_SIZE] = []
            exec_actions(packets_q, njsp, sample_rates, counters, pet_times,
                         actions_settings, streamers)

        conns = list(streamers.values()) + list(readers.values())
        for conn in conns:
            njsp.remove(conn)
        while set(conns) & set(njsp.handles):
            sleep(.1)


def exec_actions(packets_q, njsp, sample_rates, counters, pet_times, actions_settings, streamers):
    for action_id, action_settings in actions_settings.items():
        triggering = glob.TEST_TRIGGERINGS[action_id]
        # logger.debug(f'TEST_TRIGGERINGS:{glob.TEST_TRIGGERINGS}\n'
        #              f'action_id:{action_id} triggering:{triggering}')
        main_action(action_id, triggering, packets_q, pet_times, counters,
                    action_settings.get('pem', 0),
                    action_settings.get('pet', 0),
                    action_settings.get('inverse', False),
                    action_settings.get('message', None),
                    action_settings.get('address', None), njsp, sample_rates, streamers)
        if triggering == 1:
            glob.TEST_TRIGGERINGS[action_id] = -1
        elif triggering == -1:
            glob.TEST_TRIGGERINGS[action_id] = 0


def rename_packet(packet_type, content, station):
    if 'streams' == packet_type:
        stream_name = list(content.keys())[0]
        if stream_name != station:
            content[station] = content[stream_name]
            del content[stream_name]
    if 'parameters' == packet_type:
        stream_name = list(content['streams'])[0]
        if stream_name != station:
            content['streams'][station] = content['streams'][stream_name]
            del content['streams'][stream_name]
    return packet_type, content



