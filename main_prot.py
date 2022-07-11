from collections import defaultdict

import numpy as np
from obspy import UTCDateTime

import detector.misc.globals as glob
from detector.action.action_pipe import execute_action
from detector.filter_trigger.StaLtaTrigger import TriggerWrapper
from detector.filter_trigger.construct_triggers import construct_triggers
from detector.filter_trigger.rule import rule_picker
from detector.misc.globals import ActionType
from copy import deepcopy
from queue import Queue, Empty
from time import sleep

from backend.trigger_html_util import getSources, get_actions_settings, getTriggerParams, set_source_channels, get_rules_settings
from detector.action.action_process import exec_actions
from detector.misc.globals import logger
from detector.misc.misc_util import group_triggerings, to_actions_triggerings, append_test_triggerings

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
        ks = defaultdict(dict)

        triggerings = []
        rules_triggerings = []
        actions_triggerings = []

        rules_settings = get_rules_settings()

        sources = getSources()
        for station in sources:
            njsp_params = deepcopy(base_params)
            njsp_params['handshake']['client_name'] = station
            conn_data = sources[station]
            readers[station] = njsp.add_reader(conn_data['host'], int(conn_data['port']), station,
                                               njsp_params, njsp_queue)

        glob.TEST_TRIGGERINGS = {}
        actions_settings = get_actions_settings()
        for action_id in [ActionType.relay_A.value, ActionType.relay_B.value]:
            execute_action(action_id, 0, actions_settings[action_id].get('inverse', False))
        for action_id in actions_settings:
            counters[action_id] = pet_times[action_id] = 0
            glob.TEST_TRIGGERINGS[action_id] = 0
            # glob.TEST_TRIGGERINGS[action_id] = -1 if action_id in \
            #     [ActionType.relay_A.value, ActionType.relay_B.value] else 0

        triggers = construct_triggers(getTriggerParams())

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
                            station_data = content['streams'][station]
                            sample_rates[station] = station_data['sample_rate']
                            chans = list(station_data['channels'].keys())
                            set_source_channels(station, chans)
                            for chan in chans:
                                ks[station][chan] = \
                                    station_data['channels'][chan]['counts_in_volt']
                            for trigger_list in triggers[station].values():
                                for trigger in trigger_list:
                                    trigger.set_sample_rate(sample_rates[station])
                        if 'streams' == packet_type:
                            packets_q.append({packet_type: content})
                            starttime = UTCDateTime(content[station]['timestamp'])
                            channels_data = content[station]['samples']
                            # logger.debug(f'channels:{list(channels_data.keys())}')
                            for chan, bytez in channels_data.items():
                                k = ks[station][chan]
                                data = np.frombuffer(bytez, 'int').astype('float') / k
                                for trigger in triggers.get(station, {}).get(chan, []):
                                    triggerings.extend(trigger.pick(starttime, data))
                triggerings.sort()
                # process triggerings and clear after that
                for rule_id, rule_settings in rules_settings:
                    rules_triggerings.extend(rule_picker(rule_id, triggerings,
                                                         rule_settings['triggers'],
                                                         rule_settings['formula']))
                rules_triggerings.sort()
                to_actions_triggerings(rules_triggerings, rules_settings, actions_triggerings)
                append_test_triggerings(actions_triggerings, glob.TEST_TRIGGERINGS)
                actions_triggerings.sort()
                group_triggerings(triggerings, glob.USER_TRIGGERINGS, glob.LAST_TRIGGERINGS)
                group_triggerings(rules_triggerings, glob.URULES_TRIGGERINGS, glob.LAST_RTRIGGERINGS)
            except Empty:
                pass
            packets_q[:-glob.PBUF_SIZE] = []
            exec_actions(actions_triggerings, packets_q, njsp, sample_rates, counters,
                         pet_times, actions_settings, streamers)
            triggerings.clear()
            rules_triggerings.clear()
            actions_triggerings.clear()

        conns = list(streamers.values()) + list(readers.values())
        for conn in conns:
            njsp.remove(conn)
        while set(conns) & set(njsp.handles):
            sleep(.1)


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



