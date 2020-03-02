from lxml import etree
import inspect
import os
import backend
from detector.misc.globals import logger


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


#print(getChannels())
#save_pprint('<html><body>Hello<br/>World</body></html>', 'd:/temp/temp.xml')
#print(getTriggerParams())

# f = open('D:\\programming\\python\\Detector\\backend\\index.html', 'r')
# xml = f.read()
# f.close()

