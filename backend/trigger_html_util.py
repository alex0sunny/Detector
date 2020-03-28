import json

from lxml import etree
import inspect
import os
import backend
import zmq
from detector.misc.globals import logger, sources_dic, Subscription, channelsUpdater


def getHeaderMap(root):
    header_els = root.xpath('/html/body/table/tbody/tr/th')
    return {el.text: i for el, i in zip(header_els, range(100))}


def getTriggerParams():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/index.html')
    header_inds = getHeaderMap(root)
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    params_list = []
    for row in rows:
        [channel] = [el.text for el in row[header_inds['channel']].iter() if 'selected' in el.attrib]
        params_map = {'channel': channel}
        for header in header_inds.keys():
            if header not in ['channel', 'val']:
                params_map[header] = int(row[header_inds[header]].text)
        params_list.append(params_map)
    return params_list


def save_pprint(xml, file):
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml, parser).getroottree()
    header_inds = getHeaderMap(tree)
    rows = tree.xpath('/html/body/table/tbody/tr')[1:]
    for row in rows:
        row[header_inds['val']].text = '0'
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
    save_pprint(post_data_str, os.path.split(inspect.getfile(backend))[0] + '/index.html')
    clear_triggers(sockets_trigger, sockets_detrigger)
    for trigger_param in getTriggerParams():
        trigger_index = trigger_param['ind']
        if trigger_index not in sockets_trigger:
            update_sockets(trigger_index, conn_str, context, sockets_trigger, sockets_detrigger)


def post_triggers(post_data_str, sockets_trigger, sockets_detrigger):
    triggers = json.loads(post_data_str)
    triggers = {int(k): v for k, v in triggers.items()}
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
    chans = channelsUpdater.get_channels()

    # logger.debug('chans:' + str(chans) + '\ntriggers:' + str(triggers))
    json_map = {'triggers': triggers}
    # chans = ['EH1', 'EH2', 'EHN']
    if chans:
        json_map['channels'] = ' '.join(chans)
    return json_map

#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
#print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

