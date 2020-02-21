from lxml import etree
import inspect
import os
import backend


def getChannels():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/index.html')
    els = root.xpath('//option[@selected]')
    channels = [el.text for el in els]
    return channels


#print(getChannels())

