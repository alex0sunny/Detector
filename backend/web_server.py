import inspect
import json
import logging
import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os
import backend
from backend.trigger_html_util import save_pprint_trig, getTriggerParams, save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, getRuleFormulasDic, apply_sockets_rule, save_actions, \
    update_triggers_sockets

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.INFO)
logger = logging.getLogger('detector')

# curdir = './backend'
# print('list dir: ' + str(os.listdir()))
from detector.misc.globals import Port, Subscription

PORT_NUMBER = 8080

context = zmq.Context()
socket_backend = context.socket(zmq.PUB)
socket_backend.connect('tcp://localhost:' + str(Port.backend.value))

conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)

socket_channels = context.socket(zmq.SUB)
socket_channels.connect(conn_str_sub)
socket_channels.setsockopt(zmq.SUBSCRIBE, Subscription.channel.value)

chans = []

# socket_trigger = context.socket(zmq.SUB)
# socket_trigger.connect(conn_str)
# socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01011')

#trigger_params = getTriggerParams()

sockets_data_dic = {}


def create_sockets_data():
    sockets_trigger = {}
    sockets_detrigger = {}
    for trigger_param in getTriggerParams():
        update_sockets(trigger_param['ind'], conn_str_sub, context, sockets_trigger, sockets_detrigger)
    return sockets_trigger, sockets_detrigger


def get_sockets_data(session_id):
    if session_id not in sockets_data_dic:
        sockets_data_dic[session_id] = create_sockets_data()
    return sockets_data_dic[session_id]


rule_sockets_dic = {}


def create_rule_sockets():
    rule_sockets = {}
    rule_sockets_off = {}
    for rule_id in sorted(getRuleFormulasDic().keys()):
        update_sockets(rule_id, conn_str_sub, context, rule_sockets, rule_sockets_off,
                       subscription=Subscription.rule.value)
    return rule_sockets, rule_sockets_off


def get_rule_sockets(session_id):
    if session_id not in rule_sockets_dic:
        rule_sockets_dic[session_id] = create_rule_sockets()
    return rule_sockets_dic[session_id]


# This class will handles any incoming request from
# the browser
class myHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    # Handler for the GET requests
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"

        try:
            # Check the file extension required and
            # set the right mime type

            sendReply = False
            if self.path.endswith(".html"):
                logger.debug('do get, html, self.path:' + self.path)
                mimetype = 'text/html'
                sendReply = True
            if self.path.endswith(".jpg"):
                mimetype = 'image/jpg'
                sendReply = True
            if self.path.endswith(".gif"):
                mimetype = 'image/gif'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype = 'application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype = 'text/css'
                sendReply = True

            if sendReply == True:
                # Open the static file requested and send it
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type', mimetype)
                self.end_headers()
                self.wfile.write(f.read().encode())
                f.close()
            return

        except IOError:
            self.send_error(404, 'File Not Found: %s' % self.path)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        self._set_headers()

    # Handler for the POST requests
    def do_POST(self):
        # logger.debug('inside post')
        # logger.debug(self.path)
        # print(self.rfile.read())
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        post_data_str = post_data.decode()
        if self.path == '/trigger':
            # logging.info('json_map:' + str(json_map))
            json_dic = json.loads(post_data_str)
            session_id = json_dic['sessionId']
            #logger.debug('session id:' + str(session_id))
            json_triggers = json_dic['triggers']
            sockets_trigger, sockets_detrigger = get_sockets_data(session_id)
            json_map = post_triggers(json_triggers, chans, socket_channels, sockets_trigger, sockets_detrigger)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(json_map).encode())
        if self.path == '/rule':
            json_dic = json.loads(post_data_str)
            session_id = json_dic['sessionId']
            #logger.debug('session id:' + str(session_id))
            json_triggers = json_dic['triggers']
            sockets_trigger, sockets_detrigger = get_sockets_data(session_id)
            json_map = post_triggers(json_triggers, chans, socket_channels, sockets_trigger, sockets_detrigger)
            sockets_rule, sockets_rule_off = get_rule_sockets(session_id)
            rules_dic = json_dic['rules']
            rules_dic = update_rules(rules_dic, sockets_rule, sockets_rule_off)
            json_map['rules'] = rules_dic
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(json_map).encode())
        if self.path == '/initRule':
            params_list = getTriggerParams()
            trigger_ids = [params['ind'] for params in params_list]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            print('trigger ids' + str(trigger_ids))
            self.wfile.write(json.dumps(trigger_ids).encode())
        if self.path == '/apply':
            socket_backend.send(b'AP')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'apply': 1}).encode())
        if self.path == '/applyRules':
            json_dic = json.loads(post_data_str)
            session_id = json_dic['sessionId']
            html = json_dic['html']
            save_rules(html)
            sockets_rule, sockets_rule_off = get_rule_sockets(session_id)
            apply_sockets_rule(conn_str_sub, context, sockets_rule, sockets_rule_off)
            socket_backend.send(b'AP')
        if self.path == '/save':
            json_dic = json.loads(post_data_str)
            session_id = json_dic['sessionId']
            html = json_dic['html']
            save_triggers(html)
            sockets_trigger, sockets_detrigger = get_sockets_data(session_id)
            update_triggers_sockets(conn_str_sub, context, sockets_trigger, sockets_detrigger)
        if self.path == '/saveSources':
            save_sources(post_data_str)
            socket_backend.send(b'AP')
        if self.path == '/applyActions':
            save_actions(post_data_str)
            socket_backend.send(b'AP')
        if self.path == '/testActions':
            ids = json.loads(post_data_str)
            print('actions test:' + str(ids))
        if self.path == '/load':
            print('load')


try:
    # Create a web server and define the handler to manage the
    # incoming request
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print
    'Started httpserver on port ', PORT_NUMBER

    # Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    print
    '^C received, shutting down the web server'
    server.socket.close()

