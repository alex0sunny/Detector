import cgi
import inspect
import json
import logging
import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os
import backend

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

socket_channels = context.socket(zmq.SUB)
socket_channels.connect('tcp://localhost:' + str(Port.internal_resend.value))
socket_channels.subscribe(b'chan')

conn_str = 'tcp://localhost:' + str(Port.proxy.value)

# socket_trigger = context.socket(zmq.SUB)
# socket_trigger.connect(conn_str)
# socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01011')
sockets_trigger = []
sockets_detrigger = []
for trigger_index in range(3):
    socket_trigger = context.socket(zmq.SUB)
    socket_detrigger = context.socket(zmq.SUB)
    socket_trigger.connect(conn_str)
    socket_detrigger.connect(conn_str)
    trigger_index_s = '%02d' % trigger_index
    socket_trigger.setsockopt(zmq.SUBSCRIBE, b'ND01' + trigger_index_s.encode() + b'1')
    socket_detrigger.setsockopt(zmq.SUBSCRIBE, b'ND01' + trigger_index_s.encode() + b'0')
    sockets_trigger.append(socket_trigger)
    sockets_detrigger.append(socket_detrigger)

# for trigger_id in range(3):
#     socket_trigger = context.socket(zmq.SUB)
#     socket_detrigger = context.socket(zmq.SUB)
#     socket_trigger.connect(conn_str)
#     socket_detrigger.connect(conn_str)
#     trigger_index_s = '%02d' % trigger_id
#     subscription = b'ND01' + trigger_index_s.encode()
#     subscription_trigger = subscription + b'1'
#     subscription_detrigger = subscription + b'0'
#     logger.debug('subscription trigger:' + str(subscription_trigger) +
#                  '\nsubscription detrigger:' + str(subscription_detrigger))
#     socket_trigger.setsockopt(zmq.SUBSCRIBE, subscription_trigger)
#     socket_detrigger.setsockopt(zmq.SUBSCRIBE, subscription_detrigger)
#     sockets_trigger.append(socket_trigger)
#     sockets_detrigger.append(socket_detrigger)

# This class will handles any incoming request from
# the browser
class myHandler(BaseHTTPRequestHandler):

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
        obj = json.loads(post_data_str)
        print(self.path)
        if self.path == '/url':
            counter = obj['counter']
            triggers = list(map(int, obj['triggers'].split(' ')))
            for i in range(len(triggers)):
                if triggers[i]:
                    socket_target = sockets_detrigger[i]
                else:
                    socket_target = sockets_trigger[i]
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

            logging.debug('triggers:' + str(triggers))
            chans = []
            try:
                bin_data = socket_channels.recv(zmq.NOBLOCK)
                chans_bin = bin_data[4:]
                chans = [(chans_bin[i:i + 4]).decode().strip() for i in range(0, len(chans_bin), 4)]
                # if {}.update(chans_tmp) != {}.update(chans):
                #     chans = chans_tmp
                while True:
                    socket_channels.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            logger.debug('chans:' + str(chans) + '\ntriggers:' + str(triggers))
            json_map = {'counter': str(int(counter) + 1), 'triggers': ' '.join([str(trigger) for trigger in triggers])}
            #chans = ['EH1', 'EH2', 'EHN']
            if chans:
                json_map['channels'] = ' '.join(chans)
            else:
                json_map['channels'] = ''
            logging.debug('json_map:' + str(json_map))
            self.wfile.write(json.dumps(json_map).encode())
        if self.path == '/apply':
            logger.debug('object:' + str(obj) + "\nPOST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                         str(self.path), str(self.headers), post_data.decode('utf-8'))
            with open('channels.json', 'w') as f:
                json.dump(obj, f)

            socket_backend.send(b'AP')

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'apply': 1}).encode())
        if self.path == '/load':
            print('load')
            f = open(os.path.split(inspect.getfile(backend))[0] + '/channels.json')
            json_obj = json.load(f)
            f.close()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'selectedChannels': json_obj['channels']}).encode())


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
