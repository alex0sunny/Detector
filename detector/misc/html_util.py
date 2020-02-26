from lxml import etree
import inspect
import os
import backend


def getChannels():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/index.html')
    els = root.xpath('//option[@selected]')
    channels = [el.text for el in els]
    return channels


def save_pprint(xml, file):
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml, parser).getroottree()
    els = tree.xpath('/html/body/table/tbody/tr/td[3]')
    for el in els:
        el.text = '0'
    tree.write(file, pretty_print=True)


#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

