import cgi
import json
import logging
import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer
from os.path import curdir, sep
import os

# curdir = './backend'
# print('list dir: ' + str(os.listdir()))
from detector.misc.globals import Port

PORT_NUMBER = 8080

context = zmq.Context()
socket_backend = context.socket(zmq.PUB)
socket_backend.connect('tcp://localhost:%d' % Port.backend.value)

sockets_trigger = sockets_detrigger = []

conn_str = 'tcp://localhost:' + str(Port.trigger.value)
for trigger_id in range(3):
    socket_trigger = context.socket(zmq.SUB)
    socket_detrigger = context.socket(zmq.SUB)
    socket_trigger.connect(conn_str)
    socket_detrigger.connect(conn_str)
    trigger_index_s = '%02d' % trigger_id
    subscription = b'ND01' + trigger_index_s.encode()
    socket_trigger.setsockopt(zmq.SUBSCRIBE, subscription + b'1')
    socket_detrigger.setsockopt(zmq.SUBSCRIBE, subscription + b'0')
    sockets_trigger.append(socket_trigger)
    sockets_detrigger.append(socket_detrigger)

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
                print('do get, html')
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
        print('inside post')
        print(self.path)
        # print(self.rfile.read())
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        post_data_str = post_data.decode()
        obj = json.loads(post_data_str)
        if self.path == '/url':
            counter = obj['counter']
            logging.info('obj:' + str(obj))
            logging.info('triggers str:' + str(obj['triggers']))
            triggers = list(map(int, obj['triggers'].split(',')))
            if triggers[1]:
                triggers[1] = 0
            else:
                triggers[1] = 1
            logging.info('triggers:' + str(triggers))
            # print('object:' + str(obj))
            # print('counter: ' + counter)
            print('post data str:\n' + post_data_str)
            # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
            #              str(self.path), str(self.headers), post_data.decode('utf-8'))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'counter': str(int(counter) + 1), 'triggers': str(triggers)[1:-1]}).encode())
        if self.path == '/apply':
            print('object:' + str(obj))
            print("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                  str(self.path), str(self.headers), post_data.decode('utf-8'))

            socket_backend.send(b'AP')

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'apply': 1}).encode())


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
