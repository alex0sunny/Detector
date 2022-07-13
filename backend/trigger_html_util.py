import json

from lxml import etree
import inspect
import os

from lxml.etree import tostring

import backend
import zmq

from detector.filter_trigger.trigger_types import TriggerType
from detector.misc.globals import logger, Subscription, ActionType


def getHeaderDic(root):
    header_els = root.xpath('/html/body/table/tbody/tr/th')
    return {el.text: i for el, i in zip(header_els, range(100))}


def getTriggerParams():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/triggers.html')
    header_inds = getHeaderDic(root)
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    params_list = []
    for row in rows:
        params_dic = {}
        for col in ['station', 'channel', 'trigger']:
            [params_dic[col]] = \
                [el.text for el in row[header_inds[col]].iter() if 'selected' in el.attrib]
        params_dic['trigger_type'] = TriggerType[params_dic.pop('trigger')]
        params_dic['use_filter'] = 'checked' in row[header_inds['filter']][0].attrib
        for cell_name in ['name', 'sta', 'lta', 'init_level', 'stop_level', 'freqmin', 'freqmax']:
            col_type = str if cell_name == 'name' else float if cell_name in ['stop_level', 'init_level'] \
                else int
            params_dic[cell_name] = col_type(row[header_inds[cell_name]][0].get('value'))
        params_dic['ind'] = int(row[header_inds['ind']].text.strip())
        params_list.append(params_dic)
    return params_list


def get_rules_settings():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/rules.html')
    headers_dic = getHeaderDic(root)
    id_col = headers_dic['rule_id']
    formula_col = headers_dic['formula']
    actions_col = headers_dic['actions']
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    rule_dic = {}
    for row in rows:
        rule_id = int(row[id_col].text)
        rule_dic[rule_id] = {}
        formula_list = []
        for el in row[formula_col].iter():
            if 'selected' in el.attrib:
                if 'trigger_id' in el.attrib:
                    formula_list.append(int(el.attrib['trigger_id']))
                else:
                    formula_list.append(el.text)
        rule_dic[rule_id]['formula'] = formula_list
        rule_dic[rule_id]['triggers_ids'] = [formula_list[i]
                                             for i in range(0, len(formula_list), 2)]
        rule_dic[rule_id]['actions'] = [int(el.attrib['action_id'])
                                        for el in row[actions_col].iter()
                                        if 'selected' in el.attrib]
    return rule_dic


def get_sources_settings():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/sources.html')
    headers_dic = getHeaderDic(root)
    station_col = headers_dic['station']
    host_col = headers_dic['host']
    port_col = headers_dic['port']
    outport_col = headers_dic['out port']
    channels_col = headers_dic['channels']
    units_col = headers_dic['units']
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    src_dic = {}
    for row in rows:
        # logger.debug('row:' + etree.tostring(row).decode())
        station = row[station_col][0].attrib['value'].strip()
        src_dic[station] = {}
        src_dic[station]['host'] = row[host_col][0].attrib['value'].strip()
        src_dic[station]['port'] = int(row[port_col][0].attrib['value'].strip())
        src_dic[station]['out_port'] = int(row[outport_col][0].attrib['value'].strip())
        src_dic[station]['channels'] = row[channels_col].text.split(' ')
        src_dic[station]['units'] = row[units_col].text.strip()
    return src_dic


def set_source_channels(station, channels, units='V'):
    fpath = os.path.split(inspect.getfile(backend))[0] + '/sources.html'
    root = etree.parse(fpath)
    headers_dic = getHeaderDic(root)
    cell_path = "//tr[td/input/@value='" + station + "']/td"
    root.xpath(cell_path)[headers_dic['channels']].text = ' '.join(channels)
    root.xpath(cell_path)[headers_dic['units']].text = units
    root.write(fpath)


def get_action_data(action_type, root, id_col, address_col, message_col, name_col, additional_col):
    rows = root.xpath("//tr[./td/select/option[@selected]='" + action_type + "']")
    return {int(row[id_col].text): {'address': row[address_col][0].get('value'),
                                    'message': row[address_col][0].get('value'),
                                    'name': row[name_col][0].get('value'),
                                    'detrigger': 'checked' in row[additional_col][1].attrib}
            for row in rows}


def get_actions_settings():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/actions.html')
    headers_dic = getHeaderDic(root)
    id_col = headers_dic['action_id']
    address_col = headers_dic['address']
    message_col = headers_dic['message']
    name_col = headers_dic['name']
    additional_col = headers_dic['additional']
    petA = int(root.xpath("//*[@id='petA']/@value")[0])
    petB = int(root.xpath("//*[@id='petB']/@value")[0])
    infiniteA = 'checked' in root.xpath("//*[@id='infiniteA']")[0].attrib
    infiniteB = 'checked' in root.xpath("//*[@id='infiniteB']")[0].attrib
    pet_inf = 60 * 60 * 24 * 365 * 10
    if infiniteA:
        petA = pet_inf
    if infiniteB:
        petB = pet_inf
    actions_dic = {ActionType.relay_A.value: {'name': 'relayA', 'pet': petA},
                   ActionType.relay_B.value: {'name': 'relayB', 'pet': petB}}
    pem = int(root.xpath("//input[@id='PEM']/@value")[0])
    pet = int(root.xpath("//input[@id='PET']/@value")[0])
    actions_dic[ActionType.send_SIGNAL.value] = {'name': 'seedlk', 'pem': pem, 'pet': pet}
    for row in root.xpath('/html/body/table/tbody/tr')[ActionType.send_SMS.value:]:
        sms_dic = {'name': row[name_col][0].get('value'),
                   'address': row[address_col][0].get('value'),
                   'message': row[message_col][0].get('value'),
                   'inverse': 'checked' in row[additional_col][1].attrib}
        actions_dic[int(row[id_col].text)] = sms_dic
    return actions_dic


def save_pprint_trig(xml, file):
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml, parser).getroottree()
    header_inds = getHeaderDic(tree)
    rows = tree.xpath('/html/body/table/tbody/tr')[1:]
    # for row in rows:
    #     row[header_inds['val']].text = "<img src=\"img\\circle-gray.jpg\"/>"
    tree.write(file)


def save_pprint(xml, file):
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml, parser).getroottree()
    tree.write(file)


def clear_triggers(sockets_trigger, sockets_detrigger):
    for socket_cur in list(sockets_trigger.values()) + list(sockets_detrigger.values()):
        try:
            while True:
                socket_cur.recv(zmq.NOBLOCK)
        except zmq.ZMQError:
            pass


def update_sockets(trigger_index, conn_str, context, sockets_trigger, subscription=Subscription.trigger.value):
    logger.info('update sockets with ' + str(trigger_index))
    socket_trigger = context.socket(zmq.SUB)
    socket_trigger.connect(conn_str)
    trigger_index_s = '%02d' % trigger_index
    socket_trigger.setsockopt(zmq.SUBSCRIBE, subscription + trigger_index_s.encode())
    sockets_trigger[trigger_index] = socket_trigger


def save_triggers(post_data_str):
    save_pprint_trig(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/triggers.html')


def update_triggers_sockets(conn_str, context, sockets_trigger):
    # clear_triggers(sockets_trigger, sockets_detrigger)
    for trigger_param in getTriggerParams():
        trigger_index = trigger_param['ind']
        if trigger_index not in sockets_trigger:
            update_sockets(trigger_index, conn_str, context, sockets_trigger)


def save_sources(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/sources.html')


def save_rules(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/rules.html')


def save_actions(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/actions.html')


def apply_sockets_rule(conn_str, context, sockets_rule):
    # clear_triggers(sockets_rule, sockets_rule_off)
    trigger_dic = {params['ind']: params['name'] for params in getTriggerParams()}
    for rule_id in get_rules_settings().keys():
        if rule_id not in sockets_rule:
            update_sockets(rule_id, conn_str, context, sockets_rule, Subscription.rule.value)


def post_triggers(json_triggers, sockets_trigger, last_trigs=None):
    triggers = {int(k): v for k, v in json_triggers.items()}
    for i in triggers:
        if last_trigs and i in last_trigs:
            triggers[i] = last_trigs[i]
            continue
        if i in sockets_trigger:
            socket_trigger = sockets_trigger[i]
            try:
                mes = socket_trigger.recv(zmq.NOBLOCK)
                triggers[i] = int(mes[3:4])
            except zmq.ZMQError:
                pass
        else:
            #logger.warning('i ' + str(i) + ' not in triggers')
            pass
    return {'triggers': triggers}


def update_rules(json_rules, sockets_rule, last_triggerings=None):
    # logger.debug('json_rules:' + str(json_rules))
    rules = {int(k): v for k, v in json_rules.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    # logger.debug('rules:' + str(rules) + ' n of rule sockets:' + str(len(sockets_rule)))
    for i in rules:
        if last_triggerings and i in last_triggerings:
            rules[i] = last_triggerings[i]
            continue
        if i in sockets_rule:
            socket_rule = sockets_rule[i]
            try:
                mes = socket_rule.recv(zmq.NOBLOCK)
                rules[i] = int(mes[3:4])
            except zmq.ZMQError:
                pass
        else:
            logger.warning('i ' + str(i) + ' not in rules')

    return rules


# print(str(getTriggerParams()))

# print(getRuleFormulasDic())

# print(getChannels())
# save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
# print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()
