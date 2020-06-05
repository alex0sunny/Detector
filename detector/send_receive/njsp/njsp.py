import base64
import collections
import json
import queue
import select
import socket
import threading


class NJSP:
    def __init__(self, name=None):
        self.NJSP_PROTOCOL_VERIOSN = 1.0
        self.NJSP_PACKET_BUFFER_SIZE = 100
        self.NJSP_MAX_CLIENTS = 3
        self.NJSP_PROTOCOL_IDENTIFIER = b'NJSP\0\0'
        self.NJSP_HEADER_LENGTH = 8
        self.ENCODING = "ASCII"
        self.DATA_FORMAT = "signed_int32_base64"
        self.name = name
        self.error = False
        self.init_packet = None

    def decode_header(self, hdr):
        try:
            retval = int(hdr.decode(self.ENCODING), 16)
        except:
            retval = None
            self._print("Error decoding packet header")
            self.error = True
        return retval

    def decode_json(self, payload):
        try:
            retval = json.loads(payload.decode(self.ENCODING))
            if 'streams' in retval:
                for stream_name, stream in retval['streams'].items():
                    if 'samples' in stream:
                        for ch_name, ch_signal in stream['samples'].items():
                            stream['samples'].update({ch_name: base64.decodebytes(ch_signal.encode(self.ENCODING))})
        except:
            retval = None
            self._print("Error decoding json")
            self.error = True
        return retval

    def encode_hdr_and_json(self, dict_obj):
        try:
            if 'streams' in dict_obj:
                for stream_name, stream in dict_obj['streams'].items():
                    if 'samples' in stream:
                        for ch_name, ch_signal in stream['samples'].items():
                            encoded_signal = base64.encodebytes(ch_signal).decode(self.ENCODING)
                            stream['samples'].update({ch_name: encoded_signal})
            retval = json.dumps(dict_obj).encode(self.ENCODING)
            retval = format(len(retval), '08x').encode(self.ENCODING) + retval
        except:
            retval = b''
            self._print("Error encoding packet")
            self.error = True
        return retval

    def load_init_packet(self, dict_obj):
        proto_err = True
        data_fmt_err = True
        streams_err = True
        if 'parameters' in dict_obj:
            if 'protocol_version' in dict_obj['parameters']:
                if dict_obj['parameters']['protocol_version'] == self.NJSP_PROTOCOL_VERIOSN:
                    proto_err = False
            else:
                dict_obj['parameters'].update({'protocol_version': self.NJSP_PROTOCOL_VERIOSN})
                proto_err = False
            if 'welcome_msg' not in dict_obj['parameters']: dict_obj['parameters'][
                'welcome_msg'] = 'Welcome to NJSP v.1.0'
            if 'streams' in dict_obj['parameters']:
                if len(dict_obj['parameters']['streams']) > 0:
                    streams_err = False
                    for stream in dict_obj['parameters']['streams'].values():
                        if 'data_format' in stream:
                            if stream['data_format'] == self.DATA_FORMAT:
                                data_fmt_err = False
                        else:
                            data_fmt_err = False
                            stream.update({'data_format': self.DATA_FORMAT})
            else:
                data_fmt_err = False
        if proto_err: self._print("Warning! Protocol version mismatch!")
        if streams_err: self._print("Warning! No streams specified!")
        if data_fmt_err: self._print("Warning! Unsupported data format!")
        self.init_packet = dict_obj
        return proto_err or streams_err or data_fmt_err

    def _print(self, msg):
        if self.name != None:
            print("[%s] %s" % (self.name, msg))
        else:
            print(msg)


class NJSP_STREAMSERVER(NJSP):
    def __init__(self, listen_addr, init_packet, streamer_name=None):
        super().__init__(name=streamer_name)
        self.load_init_packet(init_packet)
        self.init_data = self.NJSP_PROTOCOL_IDENTIFIER + self.encode_hdr_and_json(self.init_packet)
        self.ringbuffer = collections.deque([], self.NJSP_PACKET_BUFFER_SIZE)
        self.connected_clients = dict()
        self.abort_event = threading.Event()
        self.socketserver_thread = threading.Thread(target=self.__socketserver_thread, args=([listen_addr]))
        self.socketserver_thread.start()
        self.bytes_transmitted = 0
        self.alive = True

    def kill(self):
        self.alive = False
        self.abort_event.set()
        self.socketserver_thread.join()

    def __client_add(self, socket, raddr):
        if len(self.connected_clients) > self.NJSP_MAX_CLIENTS:
            self._print("To many connections, new connection rejected")
            socket.close()
        else:
            socket.setblocking(0)
            queue = collections.deque(self.ringbuffer, self.NJSP_PACKET_BUFFER_SIZE + 3)
            queue.appendleft(self.init_data)
            self.connected_clients.update({socket: queue})
            self._print("New client %s connected" % (str(raddr)))

    def __client_remove(self, socket):
        self.connected_clients.pop(socket)
        self._print("Client disconnected")
        socket.close()

    def __client_send_packet(self, socket):
        bytes_sent = 0
        packet = self.connected_clients[socket].popleft()
        try:
            bytes_sent = socket.send(packet)
            self.bytes_transmitted += bytes_sent
        except:
            self._print("TCP socket send error")
        if bytes_sent == len(packet):
            packet = packet[bytes_sent:]
            if len(packet) > 0: self.connected_clients[socket].appendleft(packet)

    def broadcast_data(self, dict_obj):
        try:
            packet = self.encode_hdr_and_json(dict_obj)
            if packet != None:
                self.ringbuffer.append(packet)
                remove_list = list()
                for socket, queue in self.connected_clients.items():
                    if len(queue) < self.NJSP_PACKET_BUFFER_SIZE:
                        queue.append(packet)
                    else:
                        remove_list.append(socket)
                        self._print("Client %s queue is full, breaking connection" % str(socket.getpeername()))
                for socket in remove_list: self.__client_remove(socket)
        except:
            self._print("NJSP Error broadcasting packet")

    def __socketserver_thread(self, addr):
        listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listening_socket.setblocking(0)
        while True:
            try:
                listening_socket.bind(addr)
                listening_socket.listen()
                break
            except:
                self._print("TCP socket on port %s open error, retrying..." % str(addr))
            if self.abort_event.wait(10) == True:
                self._print("Server on port %s thread exited" % str(addr))
                return

        self._print("Socket server listening to %s" % str(addr))
        sockets_list = [listening_socket]
        while not self.abort_event.is_set():
            sockets_list = [listening_socket] + list(self.connected_clients.keys())
            sockets_list_to_write = list()
            for client_socket, queue in self.connected_clients.items():
                if len(queue) > 0: sockets_list_to_write.append(client_socket)
            readable, writable, exceptional = select.select(sockets_list, sockets_list_to_write, sockets_list, 0.5)
            for s in writable:
                self.__client_send_packet(s)
            for s in readable:
                if s is listening_socket:
                    self.__client_add(*s.accept())
                else:
                    self.__client_remove(s)
            for s in exceptional:
                if s is listening_socket:
                    self._print("Socket server fatal error!")
                    self.abort_event.set()
                else:
                    self.__client_remove(s)

        self.alive = False
        for s in sockets_list: s.close()
        listening_socket.close()
        self._print("NJSP Server on port %s thread exited" % str(addr))


class NJSP_STREAMREADER(NJSP):
    def __init__(self, connect_addr, reader_name=None):
        super().__init__(name=reader_name)
        self.queue = queue.Queue(self.NJSP_PACKET_BUFFER_SIZE + 2)
        self.abort_event = threading.Event()
        self.connected_event = threading.Event()
        self.disconnected_event = threading.Event()
        self.__socket = None
        self.bytes_received = 0
        self.receiver_thread = threading.Thread(target=self.__receiver_thread_fxn, args=([connect_addr]))
        self.receiver_thread.start()
        self.alive = True

    def kill(self):
        self.alive = False
        self.abort_event.set()
        self.receiver_thread.join()

    def __read_with_timeout(self, length):
        socket = self.__socket
        bytes_left = length
        data = b''
        while bytes_left > 0:
            readable, writable, exceptional = select.select([socket], list(), [socket], 0.5)
            if self.abort_event.is_set(): break
            if socket in readable:
                try:
                    new_data = socket.recv(bytes_left)
                except:
                    new_data = b''
                if new_data == b'':
                    self._print("Remote side closed connection")
                    self.error = True
                    break
                else:
                    data += new_data
                    datalen = len(new_data)
                    bytes_left -= datalen
                    self.bytes_received += datalen
            if socket in exceptional:
                self._print("Socket client fatal error!")
                self.error = True
                break
        return data

    def __read_data_packet(self):
        hdr = self.__read_with_timeout(self.NJSP_HEADER_LENGTH)
        if hdr == None: return
        data_size = self.decode_header(hdr)
        if data_size == None: return
        data = self.__read_with_timeout(data_size)
        if data == None or data == b'': return
        p = self.decode_json(data)
        if 'status' in p or 'log' in p or 'streams' in p:
            try:
                self.queue.put_nowait(p)
            except:
                self._print("Error: cannot put packet to queue")
                self.error = True

    def __read_init_packet(self):
        protocol_identifier = self.__read_with_timeout(len(self.NJSP_PROTOCOL_IDENTIFIER))
        if protocol_identifier != self.NJSP_PROTOCOL_IDENTIFIER:
            self._print("Error: protocol identifier mismatch!")
            self.error = True
        else:
            hdr = self.__read_with_timeout(self.NJSP_HEADER_LENGTH)
            data_size = self.decode_header(hdr)
            if data_size != None:
                data = self.__read_with_timeout(data_size)
                if data != b'' or data != None:
                    p = self.decode_json(data)
                    if p != None:
                        if self.load_init_packet(p) == False:
                            self.connected_event.set()
                        else:
                            self.abort_event.set()

    def __receiver_thread_fxn(self, remote_address):
        self._print("Connecting to %s" % str(remote_address))
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.__socket.connect(remote_address)
                break
            except:
                # self._print("Error connecting to remote address" )
                pass
            if self.abort_event.wait(1) == True:
                self._print("Client %s thread exited" % str(remote_address))
                self.alive = False
                return
        self._print("Connection to %s established" % str(remote_address))
        self.disconnected_event.clear()
        self.__read_init_packet()
        while not self.abort_event.is_set() and not self.error: self.__read_data_packet()
        self._print("Connection to %s closed" % str(remote_address))
        self.alive = False
        self.connected_event.clear()
        self.__socket.close()
        self.disconnected_event.set()


'''
# Server example

init_packet = {
    'parameters': {
        'streams': {
            'main': {
                'channels': {
                    'ch1': {
                        'ch_active': True
}}}}}}
streamserver = njsp.NJSP_STREAMSERVER(('localhost',12345), init_packet)
counter = 0
while counter <= 100:
    data_packet = {
        'streams': {
            'main': { 
                'samples': {
                    'ch1': counter.to_bytes(4, byteorder='little', signed=True)
    }}}}
    streamserver.broadcast_data(data_packet)
    print(counter)
    counter += 1
    time.sleep(1)
streamserver.kill()  


# Client example

streamreader = njsp.NJSP_STREAMREADER(('localhost', 12345))
streamreader.connected_event.wait()
stream_name = list(streamreader.init_packet['parameters']['streams'].keys())[0]
channel_name = list(streamreader.init_packet['parameters']['streams'][stream_name]['channels'].keys())[0]
while streamreader.connected_event.is_set():
    try: packet = streamreader.queue.get(timeout=0.5)
    except: packet = None
    if packet != None and ('streams' in packet):
        data_bytes = packet['streams'][stream_name]['samples'][channel_name]
        val = int.from_bytes(data_bytes,byteorder='little',signed=True)
        print(val)
streamreader.kill()

'''
