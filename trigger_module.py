import os, sys, json, zmq
from com_main_module import COMMON_MAIN_MODULE_CLASS
from time import sleep, time
from signal import SIGTERM
from obspy import UTCDateTime

sys.path.append(os.path.dirname(__file__))

from backend.trigger_html_util import save_pprint_trig, getTriggerParams, \
    save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, apply_sockets_rule, save_actions, \
    update_triggers_sockets, getActions, getRuleDic, getSources
from detector.misc.globals import Port, Subscription, action_names_dic0, logger

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
        p = Thread(**kwargs)
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


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
            'default_config': {'trigger_dir': '/var/lib/cloud9/trigger'}
        }

        web_ui_dir = os.path.join(os.path.dirname(__file__), "backend")
        # self._print('Initializing trigger module...')
        super().__init__(standalone, config_params, logger_config, web_ui_dir=web_ui_dir)
        config = self.get_config()
        # self._print('config:\n' + str(config) + '\n')
        sys.path.append(config['trigger_dir'])

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
            if path == 'save':
                session_id = request_dic['sessionId']
                html = request_dic['html']
                save_triggers(html)
                sockets_trigger = get_sockets_data(session_id)
                update_triggers_sockets(conn_str_sub, context, sockets_trigger)
                socket_backend.send(b'AP')
            if path == 'saveSources':
                save_sources(content.decode())
                socket_backend.send(b'AP')
            if path == 'applyActions':
                save_actions(content.decode())
                socket_backend.send(b'AP')
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
        if self.config.error:
            self.set_config(config)
            self.config.error = None

        self.message = 'Starting trigger module...'

        context = zmq.Context()
        socket_backend = context.socket(zmq.SUB)
        socket_backend.bind('tcp://*:' + str(Port.backend.value))
        socket_backend.setsockopt(zmq.SUBSCRIBE, b'AP')

        while not self.shutdown_event.is_set():
            sleep(1)
            paramsList = getTriggerParams()
            trigger_dic = {params['ind']: params['name'] for params in paramsList}
    
            kwargs_list = []
    
            action_params = getActions()
            action_names_dic = {}
            action_names_dic.update(action_names_dic0)
            sms_dic0 = action_params.get('sms', {})
            sms_dic = {sms_id: sms_dic0[sms_id]['name'] for sms_id in sms_dic0}
            action_names_dic.update(sms_dic)
            rule_dic = getRuleDic()
            rule_actions = {rule: rule_dic[rule]['actions'] for rule in rule_dic}
            action_rules = to_action_rules(rule_actions)
            send_signal_params = action_params['seedlk']
            pem = send_signal_params['pem']
            pet = send_signal_params['pet']
            rules = []
            if 3 in action_rules:
                rules = action_rules[3]
            kwargs_list += [{'target': resend, 'kwargs': {'conn_str': 'tcp://*:' + str(Port.signal_resend.value),
                                                          'rules': rules, 'pem': pem, 'pet': pet}},
                            {'target': triggers_proxy, 'kwargs': {}}]
            for action_type, send_func in {'email': send_email, 'sms': send_sms}.items():
                if action_type in action_params:
                    send_params_dic = action_params[action_type]
                    for action_id in send_params_dic:
                        rules = []
                        if action_id in action_rules:
                            rules = action_rules[action_id]
                        send_func_params = send_params_dic[action_id]
                        del send_func_params['name']
                        detrigger = send_func_params.pop('detrigger')
                        kwargs = {'action_id': action_id, 'rules': rules, 'send_func': send_func,
                                  'args': send_func_params, 'detrigger': detrigger}
                        kwargs_list.append({'target': sms_process, 'kwargs': kwargs})
            for action_id, relay_k in zip([1, 2], ['relayA', 'relayB']):
                rules = []
                if action_id in action_rules:
                    rules = action_rules[action_id]
                # print('pet:' + str(action_params['relay'][relay_k]['pet']))
                kwargs_list.append({'target': action_process,
                                    'kwargs': {'action_id': action_id, 'rules': rules, 'send_func': turn,
                                               'args': {'inverse': action_params['relay'][relay_k]['inverse']},
                                               'infinite': action_params['relay'][relay_k]['infinite'],
                                               'pet': action_params['relay'][relay_k]['pet']}})
            triggers_params = {}
            for params in paramsList:
                # params.update({'init_level': 2, 'stop_level': 1})
                trigger_params = params.copy()
                del trigger_params['name']
                trigger_params['trigger_id'] = trigger_params.pop('ind')
                station = trigger_params['station']
                channel = trigger_params['channel']
                if station not in triggers_params:
                    triggers_params[station] = {channel: []}
                elif channel not in triggers_params[station]:
                    triggers_params[station][channel] = []
                del trigger_params['station']
                del trigger_params['channel']
                triggers_params[station][channel].append(trigger_params)
                # kwargs_list.append({'target': trigger_picker, 'kwargs': trigger_params})
            for station, conn_data in getSources().items():
                kwargs = {'target': signal_receiver,
                          'kwargs': {'conn_str': 'tcp://' + conn_data['host'] + ':' + str(conn_data['port']),
                                     'station_bin': station.encode(),
                                     'triggers_params': triggers_params[station]}}
                kwargs_list.append(kwargs)
            for rule_id in sorted(rule_dic.keys()):
                formula_list = rule_dic[rule_id]['formula']
                kwargs_list.append({'target': rule_picker, 'kwargs': {'rule_id': rule_id, 'formula_list': formula_list}})
    
            threads_proc = Process(target=fps, args=(kwargs_list,))
            threads_proc.start()
    
            while not self.shutdown_event.is_set():
                poll_result = socket_backend.poll(1000)
                if poll_result:
                    socket_backend.recv()
                    break
                self.message = 'running'
        
            sleep(1)
            threads_proc.terminate()
        
        self.message = 'stopped'

        self.module_alive = False
        # self.set_config(self.get_config())
        self._print('trigger module thread exited')

