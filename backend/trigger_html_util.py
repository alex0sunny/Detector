import json

from lxml import etree
import inspect
import os
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
        [channel] = [el.text for el in row[header_inds['channel']].iter() if 'selected' in el.attrib]
        [type_str] = [el.text for el in row[header_inds['trigger']].iter() if 'selected' in el.attrib]
        trigger_type = TriggerType[type_str]
        params_map = {'channel': channel, 'trigger_type': trigger_type}
        for header in header_inds.keys():
            if header == 'station':
                params_map[header] = row[header_inds[header]].text
                continue
            if header not in ['channel', 'val', 'trigger']:
                params_map[header] = int(row[header_inds[header]].text)
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
        formula_list = [el.text for el in row[formula_col].iter() if 'selected' in el.attrib]
        while formula_list[-1] == '-' and len(formula_list) > 2:
            formula_list = formula_list[:-2]
        rule_dic[rule_id]['formula'] = formula_list
        actions_list = [int(el.text) for el in row[actions_col].iter()
                        if 'selected' in el.attrib and el.text != '-']
        rule_dic[rule_id]['actions'] = actions_list
    return rule_dic


def getSources():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/sources.html')
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    src_dic = {}
    for row in rows:
        station = row[0].text.strip()
        src_dic[station] = {}
        src_dic[station]['host'] = row[1].text.strip()
        src_dic[station]['port'] = int(row[2].text.strip())
    return src_dic


def get_action_data(action_type, root, id_col, address_col, message_col):
    rows = root.xpath("//tr[./td/select/option[@selected]='" + action_type + "']")
    return {int(row[id_col].text): {'address': row[address_col].text, 'message': row[message_col].text}
            for row in rows}


def getActions():
    actions_dic = {}
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/actions.html')
    headers_dic = getHeaderDic(root)
    id_col = headers_dic['action_id']
    address_col = headers_dic['address']
    message_col = headers_dic['message']
    for action_type in ['SMS', 'email']:
        dic = get_action_data(action_type, root, id_col, address_col, message_col)
        if dic:
            actions_dic[action_type.lower()] = dic
    pem = int(root.xpath("//input[@id='PEM']/@value")[0])
    pet = int(root.xpath("//input[@id='PET']/@value")[0])
    actions_dic['send_signal'] = {'pem': pem, 'pet': pet}
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
    for row in rows:
        row[header_inds['val']].text = '0'
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


def update_triggers_sockets(conn_str, context, sockets_trigger, sockets_detrigger):
    #clear_triggers(sockets_trigger, sockets_detrigger)
    for trigger_param in getTriggerParams():
        trigger_index = trigger_param['ind']
        if trigger_index not in sockets_trigger:
            update_sockets(trigger_index, conn_str, context, sockets_trigger, sockets_detrigger)


def save_sources(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/sources.html')


def save_rules(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/rules.html')


def save_actions(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/actions.html')


def apply_sockets_rule(conn_str, context, sockets_rule, sockets_rule_off):
    #clear_triggers(sockets_rule, sockets_rule_off)
    for rule_id in getRuleDic().keys():
        if rule_id not in sockets_rule:
            update_sockets(rule_id, conn_str, context, sockets_rule, sockets_rule_off)


def post_triggers(json_triggers, chans, socket_channels, sockets_trigger):
    triggers = {int(k): v for k, v in json_triggers.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    for i in triggers:
        if i in sockets_trigger:
            socket_trigger = sockets_trigger[i]
            try:
                mes = socket_trigger.recv(zmq.NOBLOCK)
                triggers[i] = int(mes[3:4])
                #logger.info('triggering detected, message:' + str(mes))
            except zmq.ZMQError:
                pass
        else:
            logger.warning('i ' + str(i) + ' not in triggers')

    # logging.debug('triggers:' + str(triggers))
    try:
        while True:
            chans_dic = json.loads(socket_channels.recv(zmq.NOBLOCK)[1:].decode())
            for station in chans_dic:
                chans += chans_dic[station]
    except zmq.ZMQError:
        pass
    chans = sorted(set(chans))
    #logger.debug('chans:' + str(chans))

    # logger.debug('chans:' + str(chans) + '\ntriggers:' + str(triggers))
    json_map = {'triggers': triggers}
    # chans = ['EH1', 'EH2', 'EHN']
    if chans:
        json_map['channels'] = chans
    return json_map


def update_rules(json_rules, sockets_rule):
    rules = {int(k): v for k, v in json_rules.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    #logger.debug('rules:' + str(rules) + ' n of rule sockets:' + str(len(sockets_rule)))
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


#print(str(getActions()))

#print(getRuleFormulasDic())

#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
#print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

