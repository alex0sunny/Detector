import inspect
import json
import logging
import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os
import backend
from backend.rule_html_util import post_rules
from backend.trigger_html_util import save_pprint, getTriggerParams, save_triggers, update_sockets, post_triggers

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s %(filename)s:%(lineno)d '
                           '%(message)s',
                    level=logging.INFO)
logger = logging.getLogger('detector')


# curdir = './backend'
# print('list dir: ' + str(os.listdir()))
from detector.misc.globals import Port

PORT_NUMBER = 8080

context = zmq.Context()
socket_backend = context.socket(zmq.PUB)
socket_backend.connect('tcp://localhost:' + str(Port.backend.value))

conn_str_sub = 'tcp://localhost:' + str(Port.proxy.value)

# socket_trigger = context.socket(zmq.SUB)
# socket_trigger.connect(conn_str)
# socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01011')
sockets_trigger = {}
sockets_detrigger = {}

for trigger_param in getTriggerParams():
    update_sockets(trigger_param['ind'], conn_str_sub, context, sockets_trigger, sockets_detrigger)

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
                logger.debug('do get, html')
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
        content_length = int(self.headers['Content-Length'])    # <--- Gets the size of data
        post_data = self.rfile.read(content_length)     # <--- Gets the data itself
        post_data_str = post_data.decode()
        if self.path == '/trigger':
            # logging.info('json_map:' + str(json_map))
            json_map = post_triggers(post_data_str, sockets_trigger, sockets_detrigger)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(json_map).encode())
        if self.path == '/rule':
            # logging.info('json_map:' + str(json_map))
            json_map = post_rules(post_data_str, sockets_trigger, sockets_detrigger)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(json_map).encode())
        if self.path == '/apply':
            socket_backend.send(b'AP')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'apply': 1}).encode())
        if self.path == '/applyRules':
            socket_backend.send(b'AP')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'apply': 1}).encode())
        if self.path == '/save':
            save_triggers(post_data_str, conn_str_sub, context, sockets_trigger, sockets_detrigger)
        if self.path == '/saveRules':
            save_triggers(post_data_str, conn_str_sub, context, sockets_trigger, sockets_detrigger)
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
    for socket in sockets_trigger + sockets_detrigger:
        socket.close()
