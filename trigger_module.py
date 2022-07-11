import os, sys, json, logging
from com_main_module import COMMON_MAIN_MODULE_CLASS
from time import sleep, time
from signal import SIGTERM
from obspy import UTCDateTime
from subprocess import Popen, PIPE
from queue import Queue, Empty

sys.path.append(os.path.dirname(__file__))

from backend.trigger_html_util import save_pprint_trig, getTriggerParams, \
    save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, apply_sockets_rule, save_actions, \
    update_triggers_sockets, get_actions_settings, get_rules_settings, getSources, \
    create_ref_socket, poll_ref_socket
from detector.misc.globals import Port, Subscription, action_names_dic0, CustomThread

from threading import Thread
from multiprocessing import Process

from detector.action.action_process import main_action, sms_process
from detector.action.relay_actions import turn
from detector.action.send_email import send_email
from detector.action.send_sms import send_sms
from detector.filter_trigger.rule import rule_picker
from detector.filter_trigger.rule_resender import resend
from detector.misc.misc_util import to_action_rules
from detector.send_receive.signal_receiver import signal_receiver
from detector.send_receive.triggers_proxy import triggers_proxy

last_vals = {'triggers': {}, 'rules': {}}


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
        self.message = 'Stopped'
        config = self.get_config()
        # self._print('config:\n' + str(config) + '\n')
        self.restarting = False

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
            response_dic = None
            if content and path not in ['saveSources', 'applyActions']:
                request_dic = json.loads(content.decode())

            if path == 'initTrigger':
                response_dic = getSources()
            if path == 'trigger':
                triggers = request_dic['triggers']
            if path == 'rule':
                triggers = request_dic['triggers']
                rules = request_dic['rules']
                response_dic['rules'] = rules
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
            if path == 'applyRules':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_rules(html)
                self.restarting = True
            if path == 'save':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_triggers(html)
                self.restarting = True
            if path == 'saveSources':
                save_sources(content.decode())
                self.restarting = True
            if path == 'applyActions':
                save_actions(content.decode())
                self.restarting = True
            if path == 'testActions':
                ids = request_dic['ids']
                for action_id in ids:
                    action_id_s = '%02d' % action_id
                    bin_message = Subscription.test.value + action_id_s.encode()

            if response_dic:
                content = json.dumps(response_dic).encode()
            else:
                content = b''
            return {'binary_content': content, 'code': 200,
                    'c_type': 'application/json'}

    def main(self):
        workdir = os.path.dirname(__file__)
        config = self.get_config()
        # p = Popen(['python3', workdir + '/trigger_main.py'],
        #           preexec_fn=os.setsid)
        # p = Popen(['python3', '/var/lib/cloud9/trigger/trigger_main.py'],
        #             stdout=PIPE, shell=True, preexec_fn=os.setsid)

        if self.config.error:
            self.config.set_config(config)
            self.config.error = None
        self.message = 'Starting...'
        station, conn_data = getSources().items()[0]
        njsp_params = {
            'reconnect': True,
            'reconnect_period': 30,
            'bson': True,
            'handshake': {
                'subscriptions': ['status', 'log', 'streams', 'alarms'],
                'flush_buffer': True,
                'client_name': 'TRIG'
            }
        }
        njsp_queue = Queue(100)
        host = conn_data['host']
        port = conn_data['port']
        reader_id = self.njsp.add_reader(host, port, 'TRIG', njsp_params, njsp_queue)
        check_time = UTCDateTime() + 60
        while not self.shutdown_event.is_set():
            # sleep(1)
            while not self.shutdown_event.is_set():
                if not self.njsp.isalive(reader_id):
                    self.message = 'Connecting...'
                    check_time = UTCDateTime() + 60
                self.message = 'Running'
                try:
                    packets = njsp_queue.get(timeout=1)
                    for packet_type, content in packets.items():
                        if packet_type == 'streams':
                            for stream_name, stream_data in content.items():
                                for ch_name in stream_data:
                                    stream_data[ch_name] = len(stream_data[ch_name])
                    self.logger.info('packets:\n' + str(packets))
                except Empty:
                    continue
            self.message = 'Stopped'

        # os.killpg(os.getpgid(p.pid), SIGTERM)
        self.module_alive = False
        # self._print('Main thread exited')
