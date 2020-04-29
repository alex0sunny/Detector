import inspect
import os
from time import sleep

from lxml import etree

import backend
from detector.misc.globals import logger


def turn(relay_n, on_off):
    print('inside turn, on_off:' + str(on_off))
    if relay_n == 2:
        sleep(.5)
    fname = os.path.split(inspect.getfile(backend))[0] + '/actions.html'
    root = etree.parse(fname)
    relay_cell = root.xpath("//*[@id='relay2']")[0]
    if relay_n == 1:
        relay_cell = root.xpath("//*[@id='relay1']")[0]
    relay_on = int('checked' in relay_cell.attrib)
    # actions_dic['relay1'] = int(relays[0])
    # actions_dic['relay2'] = int(relays[1])

    logger.debug('test_value before:' + str(relay_on) + ' relay_n:' + str(relay_n))
    if on_off:
        logger.info('turn the relay ' + str(relay_n) + ' on')
        relay_cell.attrib['checked'] = 'true'
    else:
        logger.info('turn the relay ' + str(relay_n) + ' off')
        relay_cell.attrib.pop('checked', None)
    fo = open(fname, 'w')
    fo.write(etree.tostring(root, pretty_print=True).decode())
    fo.close()
    logger.debug('test_value after:' + str(get_val(relay_n)) + ' relay_n:' + str(relay_n))


def get_val(relay_n):
    fname = os.path.split(inspect.getfile(backend))[0] + '/actions.html'
    root = etree.parse(fname)
    relay_cell = root.xpath("//*[@id='relay2']")[0]
    if relay_n == 1:
        relay_cell = root.xpath("//*[@id='relay1']")[0]
    relay_on = int('checked' in relay_cell.attrib)
    return relay_on
