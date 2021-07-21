import os, sys, json, zmq
from com_main_module import COMMON_MAIN_MODULE_CLASS
from time import sleep, time
from signal import SIGTERM
from obspy import UTCDateTime
from subprocess import Popen, PIPE

sys.path.append(os.path.dirname(__file__))

from backend.trigger_html_util import save_pprint_trig, getTriggerParams, \
    save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, apply_sockets_rule, save_actions, \
    update_triggers_sockets, getActions, getRuleDic, getSources
from detector.misc.globals import Port, Subscription, action_names_dic0, \
    logger, CustomThread

from threading import Thread
from multiprocessing import Process

from detector.action.action_process import action_process, sms_process
from detector.action.relay_actions import turn
from detector.action.send_email import send_email
from detector.action.send_sms import send_sms
from detector.filter_trigger.rule import rule_picker
from detector.filter_trigger.rule_resender import resend
from detector.misc.misc_util import to_action_rules
from detector.send_receive.signal_receiver import signal_receiver
from detector.send_receive.triggers_proxy import triggers_proxy


def fps(kwargs_list):
    ps = []
    for kwargs in kwargs_list:
        p = CustomThread(**kwargs)
        p.start()
        ps.append(p)
    return ps


context = zmq.Context()

socket_backend = context.socket(zmq.PUB)
socket_backend.connect('tcp://localhost:' + str(Port.backend.value))

socket_test = context.socket(zmq.PUB)
socket_test.connect('tcp://localhost:' + str(Port.multi.value))

conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)

sockets_data_dic = {}
rule_sockets_dic = {}


def create_sockets_data():
    sockets_trigger = {}
    for trigger_param in getTriggerParams():
        update_sockets(trigger_param['ind'], conn_str_sub, context, sockets_trigger)
    return sockets_trigger


def get_sockets_data(session_id):
    if session_id not in sockets_data_dic:
        sockets_data_dic[session_id] = create_sockets_data()
    return sockets_data_dic[session_id]


def create_rule_sockets():
    rule_sockets = {}
    trigger_dic = {params['ind']: params['name'] for params in getTriggerParams()}
    for rule_id in sorted(getRuleDic().keys()):
        update_sockets(rule_id, conn_str_sub, context, rule_sockets, subscription=Subscription.rule.value)
    return rule_sockets


def get_rule_sockets(session_id):
    if session_id not in rule_sockets_dic:
        rule_sockets_dic[session_id] = create_rule_sockets()
    return rule_sockets_dic[session_id]


class MAIN_MODULE_CLASS(COMMON_MAIN_MODULE_CLASS):
    def __init__(self, trigger_fxn, standalone=False):
        logger_config = {
            'log_file_name': 'trigger_module_log.txt',
            'print_to_stdout': True,
            'stdout_prefix': 'TRIGGER_MODULE'
        }
        config_params = {
            'config_file_name': 'trigger_module_cfg.json',
            #'default_config': {'trigger_dir': '/var/lib/cloud9/trigger'}
        }

        web_ui_dir = os.path.join(os.path.dirname(__file__), "backend")
        # self._print('Initializing trigger module...')
        super().__init__(standalone, config_params, logger_config, web_ui_dir=web_ui_dir)
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
                session_id = request_dic['sessionId']
                triggers = request_dic['triggers']
                sockets_trigger = get_sockets_data(session_id)
                response_dic = post_triggers(triggers, sockets_trigger)
            if path == 'rule':
                session_id = request_dic['sessionId']
                triggers = request_dic['triggers']
                sockets_trigger = get_sockets_data(session_id)
                response_dic = post_triggers(triggers, sockets_trigger)
                sockets_rule = get_rule_sockets(session_id)
                rules = request_dic['rules']
                rules = update_rules(rules, sockets_rule)
                response_dic['rules'] = rules
            if path == 'initRule':
                params_list = getTriggerParams()
                # logger.debug('params_list:' + str(params_list))
                trigger_dic = {params['ind']: params['name'] for params in params_list}
                response_dic = {'triggers': trigger_dic,
                                'actions': action_names_dic0.copy()}
                actions_dic = getActions()
                # logger.debug('getActions:' + str(actions_dic))
                sms_dic = actions_dic.get('sms', {})
                sms_dic = {sms_id: sms_dic[sms_id]['name'] for sms_id in sms_dic}
                # logger.debug('sms_dic:' + str(sms_dic) + ' json_dic:' + str(json_dic))
                response_dic['actions'].update(sms_dic)
                # logger.debug('actions_dic:' + str(json_dic['actions']))
            if path == 'apply':
                response_dic = {'apply': 1}
            if path == 'applyRules':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_rules(html)
                sockets_rule = get_rule_sockets(session_id)
                apply_sockets_rule(conn_str_sub, context, sockets_rule)
                socket_backend.send(b'AP')
                self.restarting = True
            if path == 'save':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_triggers(html)
                sockets_trigger = get_sockets_data(session_id)
                update_triggers_sockets(conn_str_sub, context, sockets_trigger)
                socket_backend.send(b'AP')
                self.restarting = True
            if path == 'saveSources':
                save_sources(content.decode())
                socket_backend.send(b'AP')
                self.restarting = True
            if path == 'applyActions':
                save_actions(content.decode())
                socket_backend.send(b'AP')
                self.restarting = True
            if path == 'testActions':
                ids = request_dic['ids']
                for action_id in ids:
                    action_id_s = '%02d' % action_id
                    bin_message = Subscription.test.value + action_id_s.encode()
                    # logger.info('send bin_message:' + str(bin_message))
                    socket_test.send(bin_message)

            if response_dic:
                content = json.dumps(response_dic).encode()
            else:
                content = b''
            return {'binary_content': content, 'code': 200,
                    'c_type': 'application/json'}

    def main(self):
        workdir = os.path.dirname(__file__)
        config = self.get_config()
        p = Popen(['python3', workdir + '/trigger_main.py'],
                  preexec_fn=os.setsid)
        # p = Popen(['python3', '/var/lib/cloud9/trigger/trigger_main.py'],
        #             stdout=PIPE, shell=True, preexec_fn=os.setsid)

        if self.config.error:
            self.set_config(config)
            self.config.error = None
        self.message = 'Starting...'
        check_time = UTCDateTime() + 60
        while not self.shutdown_event.is_set():
            # sleep(1)
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            from detector.misc.globals import Port, Subscription
            socket_sub.connect('tcp://localhost:' + str(Port.proxy.value))
            socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.signal.value)
            
            # read new packets in loop, abort if connection fails or shutdown event is set
            while not self.shutdown_event.is_set():
                if self.restarting:
                    self.errors = []
                    self.message = 'Restarting...'
                    check_time = UTCDateTime() + 60
                    self.restarting = False
                if socket_sub.poll(3000):
                    self.errors = []
                    self.message = 'Running'
                    # self._print('data received')
                    try:
                        # self._print('flush socket')
                        while True:
                            socket_sub.recv(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        # self._print('socket flushed')
                        pass
                    continue
                elif UTCDateTime() < check_time:
                    if self.message != 'Starting...':
                        self.message = 'Restarting...'
                else:
                    self.errors = ['No stations online']
                    self.message = self.errors[-1]

        os.killpg(os.getpgid(p.pid), SIGTERM)
        self.module_alive = False
        self._print('Main thread exited')

