import json
import logging
from threading import Thread
from time import sleep

import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os
import detector.misc.globals as glob

from obspy import UTCDateTime

from backend.trigger_html_util import save_pprint_trig, getTriggerParams, save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, apply_sockets_rule, save_actions, \
    update_triggers_sockets, get_actions, getRuleDic, getSources, create_ref_socket, poll_ref_socket
from detector.misc.globals import Port, Subscription, action_names_dic0, logger, CustomThread
from detector.send_receive.njsp.njsp import NJSP
from main_prot import worker

PORT_NUMBER = 8001


def trigger_module():
    njsp = NJSP(logger=logger, log_level=logging.DEBUG)
    Thread(target=worker, args=[njsp]).start()

    web_dir = os.path.dirname(__file__)
    os.chdir(web_dir)

    context = zmq.Context()
    socket_backend = context.socket(zmq.PUB)
    socket_backend.connect('tcp://localhost:' + str(Port.backend.value))

    socket_test = context.socket(zmq.PUB)
    socket_test.connect('tcp://localhost:' + str(Port.multi.value))

    conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)

    sockets_data_dic = {}
    rule_sockets_dic = {}

    last_vals = {'triggers': {}, 'rules': {}}
    ref_socket = create_ref_socket(conn_str_sub, context)

    def create_sockets_data():
        sockets_trigger = {}
        for trigger_param in getTriggerParams():
            update_sockets(trigger_param['ind'], conn_str_sub, context, sockets_trigger)
        return sockets_trigger

    def get_sockets_data(session_id):
        if session_id not in sockets_data_dic:
            if len(sockets_data_dic) > 10:
                for sid in sockets_data_dic:
                    break
                sockets_dic = sockets_data_dic.pop(sid)
                for sock in sockets_dic.values():
                    sock.close()
            sockets_data_dic[session_id] = create_sockets_data()
        return sockets_data_dic[session_id]

    def create_rule_sockets():
        rule_sockets = {}
        trigger_dic = {params['ind']: params['name'] for params in getTriggerParams()}
        for rule_id in sorted(getRuleDic().keys()):
            update_sockets(rule_id, conn_str_sub, context, rule_sockets, subscription=Subscription.rule.value)
        return rule_sockets

    def get_rule_sockets(session_id):
        if session_id not in rule_sockets_dic:
            if len(rule_sockets_dic) > 10:
                for sid in rule_sockets_dic:
                    break
                sockets_dic = rule_sockets_dic[sid]
                for sock in sockets_dic.values():
                    sock.close()
            rule_sockets_dic[session_id] = create_rule_sockets()
        return rule_sockets_dic[session_id]

    def restart_core(p):
        glob.restart = True

    # This class will handles any incoming request from
    # the browser
    class CustomHandler(BaseHTTPRequestHandler):

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

                if sendReply:
                    # Open the static file requested and send it
                    if mimetype == 'image/jpg':
                        f = open(curdir + sep + self.path, 'rb')
                    else:
                        f = open(curdir + sep + self.path)
                    self.send_response(200)
                    self.send_header('Content-type', mimetype)
                    self.end_headers()
                    if mimetype == 'image/jpg':
                        self.wfile.write(f.read())
                    else:
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
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself
            post_data_str = post_data.decode()
            # print(f'{UTCDateTime()} POST {post_data_str}')
            if self.path == '/initTrigger':
                stations_dic = getSources()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(stations_dic).encode())
                poll_ref_socket(ref_socket, last_vals)
            if self.path == '/trigger':
                # logging.info('json_map:' + str(json_map))
                json_dic = json.loads(post_data_str)
                session_id = json_dic['sessionId']
                # logger.debug('session id:' + str(session_id))
                json_triggers = json_dic['triggers']
                new_session = session_id not in sockets_data_dic
                sockets_trigger = get_sockets_data(session_id)
                if new_session:
                    # logger.debug(f'new session, last_vals:{last_vals}')
                    json_map = post_triggers(json_triggers, sockets_trigger, last_vals['triggers'])
                    # logger.debug(f'response triggerings:{json_map}')
                else:
                    json_map = post_triggers(json_triggers, sockets_trigger)
                    # logger.debug(f'response triggerings:{json_map}')
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(json_map).encode())
            if self.path == '/rule':
                json_dic = json.loads(post_data_str)
                session_id = json_dic['sessionId']
                new_session = session_id not in sockets_data_dic
                # logger.debug('session id:' + str(session_id))
                json_triggers = json_dic['triggers']
                sockets_trigger = get_sockets_data(session_id)
                if new_session:
                    json_map = post_triggers(json_triggers, sockets_trigger, last_vals['triggers'])
                else:
                    json_map = post_triggers(json_triggers, sockets_trigger)
                sockets_rule = get_rule_sockets(session_id)
                rules_dic = json_dic['rules']
                if new_session:
                    rules_dic = update_rules(rules_dic, sockets_rule, last_vals['rules'])
                else:
                    rules_dic = update_rules(rules_dic, sockets_rule)
                json_map['rules'] = rules_dic
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(json_map).encode())
            if self.path == '/initRule':
                params_list = getTriggerParams()
                logger.debug('params_list:' + str(params_list))
                trigger_dic = {params['ind']: params['name'] for params in params_list}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                # print('trigger ids' + str(trigger_ids))
                json_dic = {'triggers': trigger_dic, 'actions': action_names_dic0.copy()}
                actions_dic = get_actions()
                logger.debug('getActions:' + str(actions_dic))
                sms_dic = actions_dic.get('sms', {})
                sms_dic = {sms_id: sms_dic[sms_id]['name'] for sms_id in sms_dic}
                logger.debug('sms_dic:' + str(sms_dic) + ' json_dic:' + str(json_dic))
                json_dic['actions'].update(sms_dic)
                logger.debug('actions_dic:' + str(json_dic['actions']))
                self.wfile.write(json.dumps(json_dic).encode())
                poll_ref_socket(ref_socket, last_vals)
            if self.path == '/apply':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'apply': 1}).encode())
                glob.restart = True
            if self.path == '/applyRules':
                json_dic = json.loads(post_data_str)
                session_id = json_dic['sessionId']
                html = json_dic['html']
                save_rules(html)
                glob.restart = True
            if self.path == '/save':
                json_dic = json.loads(post_data_str)
                session_id = json_dic['sessionId']
                html = json_dic['html']
                save_triggers(html)
                glob.restart = True
            if self.path == '/saveSources':
                save_sources(post_data_str)
                socket_backend.send(b'AP')
                glob.restart = True
            if self.path == '/applyActions':
                save_actions(post_data_str)
                socket_backend.send(b'AP')
                glob.restart = True
            if self.path == '/testActions':
                test_triggerings = {int(id): v for id, v in json.loads(post_data_str).items()}
                glob.TEST_TRIGGERINGS.update(test_triggerings)
            if self.path == '/load':
                print('load')

    try:
        # Create a web server and define the handler to manage the
        # incoming request
        server = HTTPServer(('', PORT_NUMBER), CustomHandler)
        print
        'Started httpserver on port ', PORT_NUMBER
        exit(1)

        # Wait forever for incoming htto requests
        server.serve_forever()

    except KeyboardInterrupt:
        print
        '^C received, shutting down the web server'
        server.socket.close()


trigger_module()
