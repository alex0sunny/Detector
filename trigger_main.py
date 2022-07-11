import time
from multiprocessing import Process
from threading import Thread

import zmq

from backend.trigger_html_util import getTriggerParams, getSources, get_actions_settings, get_rules_settings
#from backend.web_server import web_server
from detector.action.action_process import main_action, sms_process
from detector.action.relay_actions import turn
from detector.action.send_email import send_email
from detector.action.send_sms import send_sms
from detector.filter_trigger.rule import rule_picker
from detector.filter_trigger.rule_resender import resend
from detector.misc.globals import Port, action_names_dic0
from detector.misc.misc_util import to_action_rules
from detector.send_receive.signal_receiver import signal_receiver
from detector.send_receive.triggers_proxy import triggers_proxy

use_thread = True


def fps(kwargs_list, use_thread):
    ps = []
    for kwargs in kwargs_list:
        if use_thread:
            p = Thread(**kwargs)
        else:
            p = Process(**kwargs)
        ps.append(p)
    for p in ps:
        p.start()
    return ps


if __name__ == '__main__':

    #Pool(processes=50)

    context = zmq.Context()
    socket_backend = context.socket(zmq.SUB)
    socket_backend.bind('tcp://*:' + str(Port.backend.value))
    socket_backend.setsockopt(zmq.SUBSCRIBE, b'AP')

    #Thread(target=web_server).start()

    while True:

        paramsList = getTriggerParams()
        trigger_dic = {params['ind']: params['name'] for params in paramsList}

        kwargs_list = []

        action_params = get_actions_settings()
        # print('action_params:' + str(action_params))
        # exit(1)
        action_names_dic = {}
        action_names_dic.update(action_names_dic0)
        sms_dic0 = action_params.get('sms', {})
        sms_dic = {sms_id: sms_dic0[sms_id]['name'] for sms_id in sms_dic0}
        action_names_dic.update(sms_dic)
        rule_dic = get_rules_settings()
        #print('rule_dic:' + str(rule_dic) + '\naction_names_dic:' + str(action_names_dic))
        rule_actions = {rule: rule_dic[rule]['actions'] for rule in rule_dic}
        #print('rule_actions:' + str(rule_actions))
        action_rules = to_action_rules(rule_actions)
        #print('action_rules:' + str(action_rules))
        #exit(1)
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
                    #print('send_func_params:' + str(send_func_params))
                    del send_func_params['name']
                    detrigger = send_func_params.pop('detrigger')
                    kwargs = {'action_id': action_id, 'rules': rules, 'send_func': send_func,
                              'args': send_func_params, 'detrigger': detrigger}
                    kwargs_list.append({'target': sms_process, 'kwargs': kwargs})
        for action_id, relay_k in zip([1, 2], ['relayA', 'relayB']):
            rules = []
            if action_id in action_rules:
                rules = action_rules[action_id]
            #print('pet:' + str(action_params['relay'][relay_k]['pet']))
            kwargs_list.append({'target': main_action,
                                'kwargs': {'action_id': action_id, 'rules': rules, 'send_func': turn,
                                           'args': {'inverse':  action_params['relay'][relay_k]['inverse']},
                                           'infinite': action_params['relay'][relay_k]['infinite'],
                                           'pet': action_params['relay'][relay_k]['pet']}})
        triggers_params = {}
        for params in paramsList:
            #params.update({'init_level': 2, 'stop_level': 1})
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
            #kwargs_list.append({'target': trigger_picker, 'kwargs': trigger_params})
        for station, conn_data in getSources().items():
            kwargs = {'target': signal_receiver,
                      'kwargs': {'conn_str': 'tcp://' + conn_data['host'] + ':' + str(conn_data['port']),
                                 'station_bin': station.encode(),
                                 'triggers_params': triggers_params.get(station, {})}}
            kwargs_list.append(kwargs)

        # for kwargs in kwargs_list:
        #     print('\nkwargs:\n' + str(kwargs))
        # exit(1)

        for rule_id in sorted(rule_dic.keys()):
            formula_list = rule_dic[rule_id]['formula']
            kwargs_list.append({'target': rule_picker, 'kwargs': {'rule_id': rule_id, 'formula_list': formula_list}})

        if use_thread:
            threads_proc = Process(target=fps, args=(kwargs_list, use_thread))
            threads_proc.start()
        else:
            ps = fps(kwargs_list, use_thread)

        socket_backend.recv()
        time.sleep(1)

        if use_thread:
            threads_proc.terminate()
        else:
            for p in ps:
                p.terminate()
        print('threads stopped')
    print('after break away from cycle, should exit')

