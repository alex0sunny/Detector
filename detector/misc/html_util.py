from lxml import etree
import inspect
import os
import backend
from detector.misc.globals import logger


def getTriggerParams():
    root = etree.parse(os.path.split(inspect.getfile(backend))[0] + '/index.html')
    header_els = root.xpath('/html/body/table/tbody/tr/th')
    header_inds = {el.text: i for el, i in zip(header_els, range(100)) if el.text not in ['channel', 'val']}
    rows = root.xpath('/html/body/table/tbody/tr')[1:]
    params_list = []
    parser = etree.HTMLParser(remove_blank_text=True)
    for row in rows:
        subroot_str = etree.tostring(row).decode()
        subroot = etree.fromstring(subroot_str, parser).getroottree()
        params_map = {'channel': subroot.xpath('//option[@selected]')[0].text}
        for header in header_inds.keys():
            params_map[header] = int(row[header_inds[header]].text)
        params_list.append(params_map)
    return params_list


def save_pprint(xml, file):
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml, parser).getroottree()
    els = tree.xpath('/html/body/table/tbody/tr/td[3]')
    for el in els:
        el.text = '0'
    tree.write(file)


#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

