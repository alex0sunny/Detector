import time
from multiprocessing import Process

from detector.action.action_process import action_process
from detector.action.relay_actions import turn
from detector.action.send_email import send_email
from detector.action.send_sms import send_sms
from detector.filter_trigger.rule import rule_picker
from detector.filter_trigger.StaLtaTrigger import trigger_picker
from detector.filter_trigger.rule_resender import resend
from detector.misc.globals import Port, CustomThread
from backend.trigger_html_util import getTriggerParams, getSources, getRuleFormulasDic, getActions
from detector.send_receive.signal_receiver import signal_receiver


import zmq

from detector.send_receive.triggers_proxy import triggers_proxy

use_thread = False

if __name__ == '__main__':

    context = zmq.Context()
    socket_backend = context.socket(zmq.SUB)
    socket_backend.bind('tcp://*:' + str(Port.backend.value))
    socket_backend.setsockopt(zmq.SUBSCRIBE, b'AP')

    while True:

        paramsList = getTriggerParams()

        kwargs_list = []
        for station, conn_data in getSources().items():
            kwargs = {'target': signal_receiver,
                      'kwargs': {'conn_str': 'tcp://' + conn_data['host'] + ':' + str(conn_data['port']),
                                 'station_bin': station.encode()}}
            kwargs_list.append(kwargs)

        action_params = getActions()
        send_signal_params = action_params['send_signal']
        pem = send_signal_params['pem']
        pet = send_signal_params['pet']
        kwargs_list += [{'target': resend, 'kwargs': {'conn_str': 'tcp://*:' + str(Port.signal_resend.value),
                                                      'rules': [], 'pem': pem, 'pet': pet}},
                        {'target': triggers_proxy, 'kwargs': {}}]
        for action_type, send_func in {'email': send_email, 'sms': send_sms}.items():
            send_params_dic = action_params[action_type]
            for action_id in send_params_dic:
                kwargs = {'action_id': action_id, 'send_func': send_func, 'args': send_params_dic[action_id]}
                kwargs_list.append({'target': action_process, 'kwargs': kwargs})
        for action_id in [1, 2]:
            kwargs_list.append({'target': action_process, 'kwargs': {'action_id': action_id, 'send_func': turn}})

        for params in paramsList:
            #params.update({'init_level': 2, 'stop_level': 1})
            kwargs_list.append({'target': trigger_picker, 'kwargs': params})
        # for kwargs in kwargs_list:
        #     print('\nkwargs:\n' + str(kwargs))
        # exit(1)

        formulas_dic = getRuleFormulasDic()
        for rule_id in sorted(formulas_dic.keys()):
            formula_list = formulas_dic[rule_id]
            kwargs_list.append({'target': rule_picker, 'kwargs': {'rule_id': rule_id, 'formula_list': formula_list}})

        ps = []
        for kwargs in kwargs_list:
            if use_thread:
                p = CustomThread(**kwargs)
            else:
                p = Process(**kwargs)
            ps.append(p)
        for p in ps:
            p.start()

        socket_backend.recv()
        time.sleep(1)

        for p in ps:
            p.terminate()
        print('threads stopped')
        continue
    print('after break away from cycle, should exit')
