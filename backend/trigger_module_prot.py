import json
import logging
from threading import Thread
from time import sleep

from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os
import detector.misc.globals as glob

from obspy import UTCDateTime

from backend.trigger_html_util import save_pprint_trig, getTriggerParams, save_triggers, update_sockets, post_triggers, \
    save_sources, save_rules, update_rules, apply_sockets_rule, save_actions, \
    update_triggers_sockets, get_actions_settings, get_rules_settings, get_sources_settings
from detector.misc.globals import Port, Subscription, action_names_dic0, logger, CustomThread
from detector.misc.misc_util import fill_out_triggerings
from detector.send_receive.njsp.njsp import NJSP
from main_prot import worker

PORT_NUMBER = 8001


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
            stations_dic = get_sources_settings()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stations_dic).encode())
        if self.path == '/trigger':
            # logging.info('json_map:' + str(json_map))
            json_dic = json.loads(post_data_str)
            json_triggers = json_dic['triggers']
            triggers_ids = [int(sid) for sid in json_triggers]
            triggerings_out = fill_out_triggerings(triggers_ids, glob.USER_TRIGGERINGS,
                                                   glob.LAST_TRIGGERINGS)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'triggers': triggerings_out}).encode())
        if self.path == '/rule':
            json_dic = json.loads(post_data_str)
            json_triggers = json_dic['triggers']
            triggers_ids = [int(sid) for sid in json_triggers]
            triggerings_out = fill_out_triggerings(triggers_ids, glob.USER_TRIGGERINGS,
                                                   glob.LAST_TRIGGERINGS)
            rules_dic = json_dic['rules']
            rules_ids = [int(sid) for sid in rules_dic]
            rules_out = fill_out_triggerings(rules_ids, glob.URULES_TRIGGERINGS,
                                             glob.LAST_RTRIGGERINGS)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'rules': rules_out,
                                         'triggers': triggerings_out}).encode())
        if self.path == '/initRule':
            params_list = getTriggerParams()
            logger.debug('params_list:' + str(params_list))
            trigger_dic = {params['ind']: params['name'] for params in params_list}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # print('trigger ids' + str(trigger_ids))
            json_dic = {'triggers': trigger_dic, 'actions': action_names_dic0.copy()}
            actions_dic = get_actions_settings()
            logger.debug('getActions:' + str(actions_dic))
            sms_dic = actions_dic.get('sms', {})
            sms_dic = {sms_id: sms_dic[sms_id]['name'] for sms_id in sms_dic}
            logger.debug('sms_dic:' + str(sms_dic) + ' json_dic:' + str(json_dic))
            json_dic['actions'].update(sms_dic)
            logger.debug('actions_dic:' + str(json_dic['actions']))
            self.wfile.write(json.dumps(json_dic).encode())
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
            glob.restart = True
        if self.path == '/applyActions':
            save_actions(post_data_str)
            glob.restart = True
        if self.path == '/testActions':
            test_triggerings = {int(id): v for id, v in json.loads(post_data_str).items()}
            glob.TEST_TRIGGERINGS.update(test_triggerings)
            logger.debug(f'test triggerings:{glob.TEST_TRIGGERINGS}')
        if self.path == '/load':
            print('load')


def trigger_module():
    njsp = NJSP(logger=logger, log_level=logging.DEBUG)
    Thread(target=worker, args=[njsp]).start()

    web_dir = os.path.dirname(__file__)
    os.chdir(web_dir)

    try:
        # Create a web server and define the handler to manage the
        # incoming request
        server = HTTPServer(('', PORT_NUMBER), CustomHandler)
        print
        'Started httpserver on port ', PORT_NUMBER

        # Wait forever for incoming htto requests
        server.serve_forever()

    except KeyboardInterrupt:
        print
        '^C received, shutting down the web server'
        server.socket.close()


trigger_module()
