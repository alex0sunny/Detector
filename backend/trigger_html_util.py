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


def getRuleFormulasDic():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/rules.html')
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    formulas_dic = {}
    for row in rows:
        rule_id = int(row[0].text)
        formula_list = [el.text for el in row[1].iter() if 'selected' in el.attrib]
        while formula_list[-1] == '-' and len(formula_list) > 2:
            formula_list = formula_list[:-2]
        formulas_dic[rule_id] = formula_list
    return formulas_dic


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


def update_sockets(trigger_index, conn_str, context, sockets_trigger, sockets_detrigger):
    logger.info('update sockets with ' + str(trigger_index))
    socket_trigger = context.socket(zmq.SUB)
    socket_detrigger = context.socket(zmq.SUB)
    socket_trigger.connect(conn_str)
    socket_detrigger.connect(conn_str)
    trigger_index_s = '%02d' % trigger_index
    socket_trigger.setsockopt(zmq.SUBSCRIBE,
                              Subscription.trigger.value + trigger_index_s.encode() + b'1')
    socket_detrigger.setsockopt(zmq.SUBSCRIBE,
                                Subscription.trigger.value + trigger_index_s.encode() + b'0')
    sockets_trigger[trigger_index] = socket_trigger
    sockets_detrigger[trigger_index] = socket_detrigger


def save_triggers(post_data_str, conn_str, context, sockets_trigger, sockets_detrigger):
    save_pprint_trig(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/triggers.html')
    clear_triggers(sockets_trigger, sockets_detrigger)
    for trigger_param in getTriggerParams():
        trigger_index = trigger_param['ind']
        if trigger_index not in sockets_trigger:
            update_sockets(trigger_index, conn_str, context, sockets_trigger, sockets_detrigger)


def save_sources(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/sources.html')


def save_rules(post_data_str):
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/rules.html')


def post_triggers(json_triggers, chans, socket_channels, sockets_trigger, sockets_detrigger):
    triggers = {int(k): v for k, v in json_triggers.items()}
    # logger.debug('post_data_str:' + post_data_str + '\ntriggers dic:' + str(triggers) + '\ntriggers keys:' +
    #              str(triggers.keys()))
    for i in triggers:
        # logger.debug('i:' + str(i))
        if i in sockets_trigger:
            # logger.debug('i in triggers')
            if triggers[i]:
                socket_target = sockets_detrigger[i]
                socket_non_target = sockets_trigger[i]
            else:
                socket_target = sockets_trigger[i]
                socket_non_target = sockets_detrigger[i]
            try:
                mes = socket_target.recv(zmq.NOBLOCK)
                logger.info('triggering detected, message:' + str(mes))
                if triggers[i]:
                    triggers[i] = 0
                else:
                    triggers[i] = 1
                while True:
                    socket_target.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            if triggers[i] == 0:  # clear previous triggerings
                try:
                    while True:
                        socket_non_target.recv(zmq.NOBLOCK)
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


#print(getRuleFormulasDic())

#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
#print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

