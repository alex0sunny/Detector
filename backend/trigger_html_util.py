import json

from lxml import etree
import inspect
import os

from lxml.etree import tostring

import backend
import zmq

from detector.filter_trigger.trigger_types import TriggerType
from detector.misc.globals import logger, Subscription


def getHeaderDic(root):
    header_els = root.xpath('/html/body/table/tbody/tr/th')
    return {el.text: i for el, i in zip(header_els, range(100))}


def getTriggerParams():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/triggers.html')
    header_inds = getHeaderDic(root)
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    params_list = []
    for row in rows:
        params_map = {}
        for col in ['station', 'channel', 'trigger']:
            [params_map[col]] = \
                [el.text for el in row[header_inds[col]].iter() if 'selected' in el.attrib]
        params_map['trigger_type'] = TriggerType[params_map.pop('trigger')]
        params_map['use_filter'] = 'checked' in row[header_inds['filter']][0].attrib
        for cell_name in ['name', 'sta', 'lta', 'init_level', 'stop_level', 'freqmin', 'freqmax']:
            col_type = str if cell_name == 'name' else float if cell_name in ['stop_level', 'init_level'] \
                else int
            params_map[cell_name] = col_type(row[header_inds[cell_name]][0].get('value'))
        params_map['ind'] = int(row[header_inds['ind']].text.strip())
        # [station] = [el.text for el in row[header_inds['station']].iter() if 'selected' in el.attrib]
        # [channel] = [el.text for el in row[header_inds['channel']].iter() if 'selected' in el.attrib]
        # [type_str] = [el.text for el in row[header_inds['trigger']].iter() if 'selected' in el.attrib]
        # use_filter = 'checked' in row[header_inds['filter']][0].attrib
        # trigger_type = TriggerType[type_str]
        # params_map = {'station': station, 'channel': channel, 'trigger_type': trigger_type,
        #               'use_filter': use_filter}
        # for cell_name in ['init_level', 'stop_level']:
        #     params_map[cell_name] = float(row[header_inds[cell_name]][0].get('value'))
        # excluded_headers = ['check', 'name', 'channel', 'val', 'trigger', 'filter'] + list(params_map.keys())
        # for header in header_inds.keys():
        #     if header not in excluded_headers:
        #         params_map[header] = int(row[header_inds[header]].text)
        # params_map['name'] = row[header_inds['name']].text
        params_list.append(params_map)
    return params_list


def getRuleDic():
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
                    formula_list.append(el.attrib['trigger_id'])
                else:
                    formula_list.append(el.text)
        rule_dic[rule_id]['formula'] = formula_list
        rule_dic[rule_id]['actions'] = \
            [int(el.attrib['action_id']) for el in row[actions_col].iter() if 'selected' in el.attrib]
    return rule_dic


def getSources():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/sources.html')
    headers_dic = getHeaderDic(root)
    station_col = headers_dic['station']
    host_col = headers_dic['host']
    port_col = headers_dic['port']
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
                                    'message': row[message_col][0].text.strip(),
                                    'name': row[name_col][0].get('value'),
                                    'detrigger': 'checked' in row[additional_col][1].attrib}
            for row in rows}


def getActions():
    actions_dic = {}
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/actions.html')
    headers_dic = getHeaderDic(root)
    id_col = headers_dic['action_id']
    address_col = headers_dic['address']
    message_col = headers_dic['message']
    name_col = headers_dic['name']
    additional_col = headers_dic['additional']
    for action_type in ['SMS', 'email']:
        dic = get_action_data(action_type, root, id_col, address_col, message_col, name_col, additional_col)
        # print("dic:" + str(dic))
        if dic:
            actions_dic[action_type.lower()] = dic
    pem = int(root.xpath("//input[@id='PEM']/@value")[0])
    pet = int(root.xpath("//input[@id='PET']/@value")[0])
    actions_dic['seedlk'] = {'pem': pem, 'pet': pet}
    petA = int(root.xpath("//*[@id='petA']/@value")[0])
    petB = int(root.xpath("//*[@id='petB']/@value")[0])
    infiniteA = 'checked' in root.xpath("//*[@id='infiniteA']")[0].attrib
    infiniteB = 'checked' in root.xpath("//*[@id='infiniteB']")[0].attrib
    inverseA = 'checked' in root.xpath("//*[@id='inverseA']")[0].attrib
    inverseB = 'checked' in root.xpath("//*[@id='inverseB']")[0].attrib
    actions_dic['relay'] = {'relayA': {'infinite': infiniteA, 'pet': petA, 'inverse': inverseA},
                            'relayB': {'infinite': infiniteB, 'pet': petB, 'inverse': inverseB}}
    # relay1cell = root.xpath("//*[@id='relay1']")[0]
    # relay2cell = root.xpath("//*[@id='relay2']")[0]
    # relays = ['checked' in releCell.attrib for releCell in [relay1cell, relay2cell]]
    # actions_dic['relay1'] = int(relays[0])
    # actions_dic['relay2'] = int(relays[1])
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
    for rule_id in getRuleDic().keys():
        if rule_id not in sockets_rule:
            update_sockets(rule_id, conn_str, context, sockets_rule, Subscription.rule.value)


def post_triggers(json_triggers, sockets_trigger):
    triggers = {int(k): v for k, v in json_triggers.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    for i in triggers:
        if i in sockets_trigger:
            socket_trigger = sockets_trigger[i]
            try:
                mes = socket_trigger.recv(zmq.NOBLOCK)
                triggers[i] = int(mes[3:4])
                # logger.info('triggering detected, message:' + str(mes))
            except zmq.ZMQError:
                pass
        else:
            logger.warning('i ' + str(i) + ' not in triggers')

    return {'triggers': triggers}


def update_rules(json_rules, sockets_rule):
    # logger.debug('json_rules:' + str(json_rules))
    rules = {int(k): v for k, v in json_rules.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    # logger.debug('rules:' + str(rules) + ' n of rule sockets:' + str(len(sockets_rule)))
    for i in rules:
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

# print(str(getActions()))

# print(getRuleFormulasDic())

# print(getChannels())
# save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
# print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()
