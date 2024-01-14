import json
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from queue import Queue, Empty
from time import sleep

import numpy as np
from com_main_module import COMMON_MAIN_MODULE_CLASS
from obspy import UTCDateTime

sys.path.append(os.path.dirname(__file__))

import detector.misc.globals as glob
from detector.action.action_pipe import execute_action
from detector.filter_trigger.construct_triggers import construct_triggers
from main_prot import rename_packet

from backend.trigger_html_util import getTriggerParams, \
    save_triggers, save_sources, save_rules, save_actions, \
    get_actions_settings, get_rules_settings, get_sources_settings, set_source_channels
from detector.misc.globals import action_names_dic0, ActionType, ConnState

from detector.action.action_process import exec_actions
from detector.filter_trigger.rule import rule_picker
from detector.misc.misc_util import fill_out_triggerings, append_test_triggerings, \
    to_actions_triggerings, group_triggerings


class MAIN_MODULE_CLASS(COMMON_MAIN_MODULE_CLASS):

    def __init__(self, njsp, trigger_fxn, standalone=False):

        logger_config = {
            'logger_name': 'trigger',
            'file_name': 'trigger',
            'files_dir': '/media/sdcard/logs',
            'file_level': logging.DEBUG,
            'console_level': logging.DEBUG,  # if standalone else logging.WARN,
            'console_name': None if standalone else 'TRIGGER_MODULE'
        }

        config_params = {
            'config_file_name': 'trigger_module_cfg.json',
            # 'default_config': {'trigger_dir': '/var/lib/cloud9/trigger'}
        }

        web_ui_dir = os.path.join(os.path.dirname(__file__), "backend")
        # self._print('Initializing trigger module...')
        super().__init__(standalone, config_params, njsp, logger_config, web_ui_dir=web_ui_dir)
        self.message = 'starting...'
        config = self.get_config()
        # self._print('config:\n' + str(config) + '\n')
        self.restarting = False
        self.njsp = njsp

    def custom_web_ui_request(self, in_data):
        path = in_data['path']  # .split('?')
        for ext in ['jpg', 'png', 'ico', 'gif']:
            if path.endswith('.' + ext):
                f = open(self.web_ui_dir + os.sep + path, 'rb')
                data = f.read()
                f.close()
                return {'binary_content': data, 'code': 200,
                        'c_type': 'image/' + ext}
        if in_data['type'] == 'post':
            content = in_data['binary_content']
            response_dic = {}
            if content and path not in ['saveSources', 'applyActions']:
                request_dic = json.loads(content.decode())

            if path == 'initTrigger':
                response_dic = get_sources_settings()
            if path == 'trigger':
                triggers = request_dic['triggers']
                triggers_ids = [int(sid) for sid in triggers]
                triggerings_out = fill_out_triggerings(triggers_ids, glob.USER_TRIGGERINGS,
                                                       glob.LAST_TRIGGERINGS)
                response_dic['triggers'] = triggerings_out
            if path == 'rule':
                triggers = request_dic['triggers']
                triggers_ids = [int(sid) for sid in triggers]
                triggerings_out = fill_out_triggerings(triggers_ids, glob.USER_TRIGGERINGS,
                                                       glob.LAST_TRIGGERINGS)
                response_dic['triggers'] = triggerings_out
                rules = request_dic['rules']
                rules_ids = [int(sid) for sid in rules]
                rules_out = fill_out_triggerings(rules_ids, glob.URULES_TRIGGERINGS,
                                                 glob.LAST_RTRIGGERINGS)
                response_dic['rules'] = rules_out
            if path == 'initRule':
                params_list = getTriggerParams()
                trigger_dic = {params['ind']: params['name'] for params in params_list}
                response_dic = {'triggers': trigger_dic,
                                'actions': action_names_dic0.copy()}
                actions_dic = get_actions_settings()
                sms_dic = actions_dic.get('sms', {})
                sms_dic = {sms_id: sms_dic[sms_id]['name'] for sms_id in sms_dic}
                response_dic['actions'].update(sms_dic)
            if path == 'apply':
                response_dic = {'apply': 1}
                glob.restart = True
            if path == 'applyRules':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_rules(html)
                self.restarting = glob.restart = True
            if path == 'save':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_triggers(html)
                self.restarting = glob.restart = True
            if path == 'saveSources':
                save_sources(content.decode())
                self.restarting = glob.restart = True
            if path == 'applyActions':
                save_actions(content.decode())
                self.restarting = glob.restart = True
            if path == 'testActions':
                test_triggerings = {int(aid): v for aid, v in request_dic.items()}
                glob.TEST_TRIGGERINGS.update(test_triggerings)

            if response_dic:
                content = json.dumps(response_dic).encode()
            else:
                content = b''
            return {'binary_content': content, 'code': 200,
                    'c_type': 'application/json'}

    def main(self):
        workdir = os.path.dirname(__file__)
        config = self.get_config()

        if self.config.error:
            self.config.set_config(config)
            self.config.error = None
        logger = glob.logger = self.logger
        self.message = 'Starting...'
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

        while not self.shutdown_event.is_set():
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

            sources = get_sources_settings()
            for station in sources:
                njsp_params = deepcopy(base_params)
                njsp_params['handshake']['client_name'] = station
                conn_data = sources[station]
                readers[station] = self.njsp.add_reader(conn_data['host'], int(conn_data['port']), station,
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

            check_time = UTCDateTime()
            glob.CONN_STATE = ConnState.CONNECTING
            while not glob.restart and not self.shutdown_event.is_set():
                if glob.CONN_STATE != ConnState.CONNECTED:
                    self.message = glob.CONN_STATE.name.lower()
                elif any(glob.LAST_RTRIGGERINGS.values()):
                    self.message = 'TRIGGERED'
                else:
                    self.message = 'running'
                cur_time = UTCDateTime()
                try:
                    packets_data = njsp_queue.get(timeout=1)
                    check_time = cur_time
                    glob.CONN_STATE = ConnState.CONNECTED
                    for conn_name, dev_packets in packets_data.items():
                        station = conn_name.split(':')[1]
                        for packet_type, content in dev_packets.items():
                            packet_type, content = rename_packet(packet_type, content, station)
                            if 'parameters' == packet_type and station not in streamers:
                                streamer_params = {'init_packet': {'parameters': content.copy()},
                                                   'ringbuffer_size': 10}
                                streamers[station] = self.njsp.add_streamer('', sources[station]['out_port'],
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
                    for rule_id, rule_settings in rules_settings.items():
                        rules_triggerings.extend(rule_picker(rule_id, triggerings,
                                                             rule_settings['triggers_ids'],
                                                             rule_settings['formula']))
                    rules_triggerings.sort()
                    # logger.debug(f'rules_triggerings:{rules_triggerings}')
                    to_actions_triggerings(rules_triggerings, rules_settings, actions_triggerings)
                    actions_triggerings.sort()
                    group_triggerings(triggerings, glob.USER_TRIGGERINGS, glob.LAST_TRIGGERINGS)
                    group_triggerings(rules_triggerings, glob.URULES_TRIGGERINGS, glob.LAST_RTRIGGERINGS)
                except Empty:
                    if cur_time > check_time + 10:
                        glob.CONN_STATE = ConnState.NO_CONNECTION
                packets_q[:-glob.PBUF_SIZE] = []
                if any(glob.TEST_TRIGGERINGS.values()):
                    logger.debug(f'test triggerings:{glob.TEST_TRIGGERINGS}')
                append_test_triggerings(actions_triggerings, glob.TEST_TRIGGERINGS)
                # logger.debug(f'actions triggerings:{actions_triggerings}')
                exec_actions(actions_triggerings, packets_q, self.njsp, sample_rates, counters,
                             pet_times, actions_settings, streamers)
                triggerings.clear()
                rules_triggerings.clear()
                actions_triggerings.clear()

            conns = list(streamers.values()) + list(readers.values())
            for conn in conns:
                self.njsp.remove(conn)
            while set(conns) & set(self.njsp.handles):
                sleep(.1)

        self.module_alive = False
        self.message = 'Stopped'
        # self._print('Main thread exited')
