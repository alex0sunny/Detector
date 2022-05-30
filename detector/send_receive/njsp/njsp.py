import json, base64, bson, threading, socket, select, collections, queue, time, logging

class NJSP_LOGGER_ADAPTER(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        self.set_prefix(prefix)
        super().__init__(logger, False)
        
    def set_prefix(self, prefix):
        if prefix != None: self.prefix = '[%s] '%prefix
        else: self.prefix = ''
        
    def process(self, msg, kwargs):
        return (self.prefix + msg), kwargs

class NJSP:
    def __init__(self, logger = None, name = None):
        if name != None: 
            if logger == None: self.logger = NJSP_LOGGER_ADAPTER(logging.getLogger('NJSP'), name)
            else: self.logger = NJSP_LOGGER_ADAPTER(logger, 'NJSP:%s'%name)
        else: 
            if logger == None: self.logger = logging.getLogger('NJSP')
            else: self.logger = NJSP_LOGGER_ADAPTER(logger, 'NJSP')
        logging.basicConfig(level=logging.DEBUG)
        self.NJSP_PROTOCOL_VERIOSN = 2.0
        self.NJSP_PACKET_BUFFER_SIZE = 100
        self.NJSP_MAX_CLIENTS = 5
        self.NJSP_PROTOCOL_ID = b'NBJSP\0'
        self.NJSP_HEADER_LENGTH = 8
        self.ENCODING = "ASCII"
        self.DATA_FORMAT = "signed_int32"
        self.error = False
        self.init_packet = None
        
    def decode_header(self, hdr):
        try:
            retval = int(hdr.decode(self.ENCODING), 16)
        except:
            retval = None
            self.logger.error("Error decoding packet header %s" %str(hdr))
            self.error = True
        return retval
        
    def decode_json(self, payload):
        try: 
            retval = json.loads(payload.decode(self.ENCODING))
            #if 'status' in retval: print(json.dumps(retval, indent=4, sort_keys=True))
            if 'streams' in retval:
                for stream_name, stream in retval['streams'].items():
                    if 'samples' in stream:
                        for ch_name, ch_signal in stream['samples'].items():
                            stream['samples'].update({ch_name:base64.decodebytes(ch_signal.encode(self.ENCODING))})
        except:
            retval = None
            self.logger.error("Error decoding json")
            self.error = True
        return retval
        
    def encode_hdr_and_json(self, dict_obj):
        try:
            if 'streams' in dict_obj:
                for stream_name, stream in dict_obj['streams'].items():
                    if 'samples' in stream:
                        for ch_name, ch_signal in stream['samples'].items():
                            encoded_signal = base64.encodebytes(ch_signal).decode(self.ENCODING)
                            stream['samples'].update({ch_name:encoded_signal})
            retval = json.dumps(dict_obj).encode(self.ENCODING)
            retval = format(len(retval), '08x').encode(self.ENCODING) + retval
        except:
            retval = b''
            self.logger.error("Error encoding packet")
            self.error = True
        return retval
        
        
    def split_to_datatypes_and_encode(self, dict_obj):
        out_packet = dict()
        for data_type, payload in dict_obj.items():
            out_packet.update({data_type:self.encode_hdr_and_json({data_type:payload})})
        return out_packet
        
    def check_init_packet(self, dict_obj):
        proto_err = True
        data_fmt_err = False
        streams_err = True
        sample_rate_err = False
        if 'parameters' in dict_obj:
            if 'protocol_version' in dict_obj['parameters']:
                if dict_obj['parameters']['protocol_version'] == self.NJSP_PROTOCOL_VERIOSN: 
                    proto_err = False
            else: 
                dict_obj['parameters'].update({'protocol_version':self.NJSP_PROTOCOL_VERIOSN})
                proto_err = False
            if 'welcome_msg' not in dict_obj['parameters']: 
                dict_obj['parameters']['welcome_msg'] = 'Hello! This is NJSP v%.2f'%self.NJSP_PROTOCOL_VERIOSN
            if 'streams'  in dict_obj['parameters']:
                if len(dict_obj['parameters']['streams']) > 0: 
                    streams_err = False
                    for stream in dict_obj['parameters']['streams'].values():
                        if 'data_format' in stream:
                            if stream['data_format'] != self.DATA_FORMAT: data_fmt_err = True
                        else:
                            stream.update({'data_format':self.DATA_FORMAT})
                        if 'sample_rate' not in stream: sample_rate_err = True
            else: data_fmt_err = False
        if proto_err: self.logger.error("Error! Protocol version mismatch!")
        if streams_err: self.logger.error("Error! No streams specified!")
        if data_fmt_err: self.logger.error("Error! Unsupported data format!")
        if sample_rate_err: self.logger.error("Error! Sample rate not specified")
        return proto_err or streams_err or data_fmt_err or sample_rate_err
        
    def create_handshake_packet(self, user_params):
        if user_params == None: user_params = dict()
        user_params.update({'protocol_version': self.NJSP_PROTOCOL_VERIOSN})
        if 'subscriptions' not in user_params: user_params.update({'subscriptions': ['status','log','streams','alarms']})
        if 'flush_buffer' not in user_params: user_params.update({'flush_buffer': True})
        if 'client_name' not in user_params: user_params.update({'client_name': socket.gethostname()})
        return {'handshake': user_params}

class NBJSP(NJSP):
    def decode_json(self, payload):
        try: 
            retval = bson.loads(payload)
        except:
            retval = None
            self.logger.error("Error decoding json")
            self.error = True
        return retval
        
    def encode_hdr_and_json(self, dict_obj):
        try:
            retval = bson.dumps(dict_obj)
            retval = format(len(retval), '08x').encode(self.ENCODING) + retval
        except:
            retval = b''
            self.logger.error("Error encoding packet")
            self.error = True
        return retval

class NJSP_CLIENT_HANDLER:
    def __init__(self, logger, streamserver, socket_obj, client_addr):
        self.server = streamserver
        self.addr = '%s:%s'%(client_addr[0],client_addr[1])
        self.logger = NJSP_LOGGER_ADAPTER(logger, self.addr)
        self.__socket = socket_obj
        self.__socket.setblocking(0)
        self.queue = collections.deque([self.server.init_data_bin], self.server.NJSP_PACKET_BUFFER_SIZE*2)
        self.logger.info("Socket opened, handshaking...")
        self.subscriptions = list()
        self.queue_ready = False
        self.client_alive = True
        self.recevice_statemachine = {
            'state':'waiting_id',
            'recvd_bytes':b'',
            'bytes_left':len(self.server.NJSP_PROTOCOL_ID)
        }
        
    def __new_packet(self, in_packet):
        retval = 'OK'
        for data_type, payload in in_packet.items():
            if data_type in self.subscriptions:
                if len(self.queue) >= (self.queue.maxlen-1):
                    retval = 'error'
                    self.logger.error("Queue max length reached, breaking connection")
                    self.client_alive = False
                    self.queue_ready = False
                    break
                else: self.queue.append(payload) 
        return retval
        
    def new_packet(self, in_packet):
        if self.queue_ready: return self.__new_packet(in_packet)
        else: return 'OK'
        
    def __process_handshake_packet(self, incoming_payload):
        retval = 'error'
        dict_obj = self.server.decode_json(incoming_payload)
        if dict_obj != None and 'handshake' in dict_obj: 
            client_params = dict_obj['handshake']
            if 'protocol_version' in client_params:
                if client_params['protocol_version'] == self.server.NJSP_PROTOCOL_VERIOSN: 
                    retval = 'OK'
                    if 'client_name' in client_params: 
                        self.logger.debug("Client name is %s"%client_params['client_name'])
                        self.logger.prefix_msg = '%s:%s'%(self.addr, client_params['client_name'])
                    if 'subscriptions' in client_params: 
                        self.subscriptions = client_params['subscriptions']
                        self.logger.info("Subscriptions: %s"%str(self.subscriptions))
                    if 'flush_buffer' in client_params:
                        if client_params['flush_buffer']:
                            self.logger.debug("Flushing buffer...")
                            self.queue_ready = True
                            #with self.server.ringbuffer_lock:
                            for packet in list(self.server.ringbuffer): self.__new_packet(packet)
                        else: self.queue_ready = True
                else: self.logger.error("Protocol version mismatch (client %d server %d)!"\
                                    %(client_params['protocol_version'],self.server.NJSP_PROTOCOL_VERIOSN))
        return retval
    
    def __recv_stm_worker(self, new_bytes):
        retval = 'error'
        if self.recevice_statemachine['bytes_left'] > len(new_bytes): bytes_consumed = new_bytes
        else: bytes_consumed = new_bytes[:self.recevice_statemachine['bytes_left']]
        self.recevice_statemachine['recvd_bytes'] += bytes_consumed
        self.recevice_statemachine['bytes_left'] -= len(bytes_consumed)
        
        if self.recevice_statemachine['bytes_left'] > 0: 
            retval = 'OK'
            
        elif self.recevice_statemachine['state'] == 'waiting_id':
            if self.recevice_statemachine['recvd_bytes'] != self.server.NJSP_PROTOCOL_ID:
                self.logger.error("Error: incorrect protocol id (%s)!"%self.recevice_statemachine['recvd_bytes'])
            else:
                self.recevice_statemachine['state'] = 'waiting_hdr'
                self.recevice_statemachine['recvd_bytes'] = b''
                self.recevice_statemachine['bytes_left'] = self.server.NJSP_HEADER_LENGTH
                retval = 'OK'

        elif self.recevice_statemachine['state'] == 'waiting_hdr':
            payload_size = self.server.decode_header(self.recevice_statemachine['recvd_bytes'])
            if payload_size != None:
                self.recevice_statemachine['state'] = 'waiting_payload'
                self.recevice_statemachine['recvd_bytes'] = b''
                self.recevice_statemachine['bytes_left'] = payload_size
                retval = 'OK'
            
        elif self.recevice_statemachine['state'] == 'waiting_payload':
            retval = self.__process_handshake_packet(self.recevice_statemachine['recvd_bytes'])
            self.recevice_statemachine['state'] = 'handshake_ok'
            self.recevice_statemachine['recvd_bytes'] = b''
            self.recevice_statemachine['bytes_left'] = 0
            
        else: self.logger.error("Received unexpected data from client, disconnecting: %s"%str(new_bytes))
            
        return new_bytes[len(bytes_consumed):], retval
    
    def receive_packet(self):
        retval = 'OK'; bytes = b''
        try: bytes = self.__socket.recv(4096)
        except: retval = 'error'
        if len(bytes) == 0: retval = 'error'
        #else: self.logger.debug("Received %d bytes from client"%len(bytes))
        while len(bytes) > 0 and retval == 'OK': bytes, retval = self.__recv_stm_worker(bytes)
        return retval
        
    def send_packet(self):
        bytes_sent = 0
        packet_bin = self.queue.popleft()
        try: bytes_sent = self.__socket.send(packet_bin)
        except: self.logger.error("TCP socket send error")
        bytes_left = len(packet_bin) - bytes_sent
        if bytes_left > 0: 
            self.queue.appendleft(packet_bin[bytes_sent:])
            self.logger.debug('Sent %db, putting %db back to queue'%(bytes_sent, bytes_left))

    def disconnect(self):
        self.__socket.close()
        self.logger.warning("Client disconnected")


class NJSP_STREAMSERVER(NBJSP):
    def __init__(self, listen_addr, init_packet, logger = None, streamer_name = None):
        super().__init__(logger = logger, name = streamer_name)
        if self.check_init_packet(init_packet) == False: self.init_packet = init_packet
        self.init_data_bin = self.NJSP_PROTOCOL_ID + self.encode_hdr_and_json(self.init_packet)
        self.ringbuffer = collections.deque([], self.NJSP_PACKET_BUFFER_SIZE)
        self.connected_clients = dict()
        self.global_lock = threading.Lock()
        self.kill_event = threading.Event()
        self.alive = True
        self.socketserver_thread = threading.Thread(target=self.__socketserver_thread_fxn, args=([listen_addr]))
        self.socketserver_thread.name = 'njsp_srv'
        self.socketserver_thread.start()
            
    def kill(self):
        self.kill_event.set()
        self.socketserver_thread.join()
        
    def broadcast_data(self, dict_obj):
        if dict_obj == None or self.alive == False: return
        packet = self.split_to_datatypes_and_encode(dict_obj)
        if len(packet) > 0:
            with self.global_lock: 
                self.ringbuffer.append(packet)
                for client in self.connected_clients.values(): client.new_packet(packet)

    def __socketserver_main_loop(self, listening_socket):
        listen_socket_error = False
        while not (self.kill_event.is_set() or listen_socket_error):
            
            read_list = list(self.connected_clients.keys())
            write_list = list()
            remove_list = list()
            
            if len(self.connected_clients) < self.NJSP_MAX_CLIENTS: read_list.append(listening_socket)
            
            for client_socket, client_handler in self.connected_clients.items(): 
                if not client_handler.client_alive: remove_list.append(client_socket)
                if len(client_handler.queue) > 0: write_list.append(client_socket)
                
            with self.global_lock:
                for s in remove_list: self.connected_clients.pop(s).disconnect()

            r, w, e = select.select(read_list, write_list, read_list, 0.5)
            
            with self.global_lock:
                for s in w:
                    if s in self.connected_clients: self.connected_clients[s].send_packet()
                    else: s.close()
                for s in r:
                    if s is listening_socket: 
                        socket_obj, raddr = s.accept()
                        new_client = NJSP_CLIENT_HANDLER(self.logger, self, socket_obj, raddr)
                        self.connected_clients.update({socket_obj:new_client})
                    elif s in self.connected_clients: 
                        result = self.connected_clients[s].receive_packet()
                        if result != 'OK': self.connected_clients.pop(s).disconnect()
                    else: s.close()
                for s in e:
                    if s is listening_socket:
                        self.logger.critical("Socket server fatal error!") 
                        listen_socket_error = True
                    else: 
                        self.connected_clients.pop(s).disconnect()
                    
        # close all connections with command abort
        start_time = time.monotonic()
        sockets_list = list(self.connected_clients.keys())
        abort_bin = self.encode_hdr_and_json({'abort':'Server stopped'})
        
        #TODO: if client queue has unsent part of some packet, send it first
        for s in sockets_list: 
            try: s.send(abort_bin)
            except: pass
        
        while len(sockets_list) > 0:
            r, w, e = select.select(sockets_list, list(), list(), 1)
            for s in r:
                try: bytes = s.recv(4096)
                except: bytes = b''
                if bytes == b'':
                    s.close()
                    sockets_list.remove(s)
            if time.monotonic() - start_time > 1: break
        if len(sockets_list) > 0: self.logger.warning("%d clients were not disconnected properly"%len(sockets_list))
        else: self.logger.debug("All clients disconnected in %dms"%((time.monotonic() - start_time)*1000))
        
        listening_socket.close()

    def __socketserver_thread_fxn(self, addr):
        listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listening_socket.setblocking(0)
        socket_opened = False
        while not socket_opened:
            try:
                listening_socket.bind(addr)
                listening_socket.listen()
                self.logger.info("Listening to %s" %str(addr))
                socket_opened = True
            except: self.logger.error("TCP socket on port %s open error, retrying..." %str(addr))
            if self.kill_event.wait(10) == True: break
        
        if socket_opened: self.__socketserver_main_loop(listening_socket)
            
        self.kill_event.wait() # loop until killed
        self.logger.info("Server thread (port %s) exited" %str(addr))
        

class NJSP_READER_HANDLER:
    def __init__(self, server, name, addr, params):
        self.server = server
        self.name = name
        self.logger = NJSP_LOGGER_ADAPTER(server.logger, '%s:%d'%(addr[0],addr[1]))
        self.logger.warning("New client added, establishing connection..")
        self.addr = addr
        self.params = params
        self.socket_opened = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(0)
        self.handshake_bytes = self.server.NJSP_PROTOCOL_ID
        self.handshake_bytes += self.server.encode_hdr_and_json(self.server.create_handshake_packet(params))
        self.state = 'establishing_connection'
        self.__send_state_update('connecting')
        
    def __send_state_update(self, state):
        self.server.queue.put_nowait({self.name:{'connection_state':state}})

    def try_to_connect(self):
        try:
            self.socket.settimeout(0.1)
            self.socket.connect(self.addr)
            self.socket.settimeout(0)
            self.logger.debug("Connection established, handshaking..")
            self.state = 'sending_handshake'
            self.socket_opened = True
            self.bytes_to_write = self.handshake_bytes
            self.__rearm_recv_stm('waiting_id', len(self.server.NJSP_PROTOCOL_ID))
        except:
            pass
            
    def __packet_received(self, packet):
        if 'abort' in packet:
            self.logger.warning("Received abort command, disconnecting...")
            self.disconnect(reconnect = True)
        else:
            if self.state == 'waiting_init_packet':
                if self.server.check_init_packet(packet) == False: 
                    self.init_packet = packet
                    self.state = 'receiving_data'
                    self.__send_state_update('connected')
                    self.logger.warning("Connected!")
                else: 
                    self.logger.critical("Error reading init packet")
                    self.disconnect(reconnect = False)
            out_packet = {self.name:packet}
            try: self.server.queue.put_nowait(out_packet)
            except: self.logger.error("Error putting packet to queue")
        
    def __rearm_recv_stm(self, state, length):
        self.recevice_statemachine = {
            'state':state,
            'recvd_bytes':b'',
            'bytes_left':length
        }
                    
    def __recv_stm_worker(self):
        retval = 'error'
            
        if self.recevice_statemachine['state'] == 'waiting_id':
            if self.recevice_statemachine['recvd_bytes'] != self.server.NJSP_PROTOCOL_ID:
                self.logger.error("Incorrect protocol id (%s)!"%str(self.recevice_statemachine['recvd_bytes']))
            else:
                self.__rearm_recv_stm('waiting_hdr', self.server.NJSP_HEADER_LENGTH)
                retval = 'OK'

        elif self.recevice_statemachine['state'] == 'waiting_hdr':
            payload_size = self.server.decode_header(self.recevice_statemachine['recvd_bytes'])
            if payload_size != None:
                self.__rearm_recv_stm('waiting_payload', payload_size)
                retval = 'OK'
            
        elif self.recevice_statemachine['state'] == 'waiting_payload':
            packet = self.server.decode_json(self.recevice_statemachine['recvd_bytes'])
            if packet != None: 
                self.__packet_received(packet)
                self.__rearm_recv_stm('waiting_hdr', self.server.NJSP_HEADER_LENGTH)
                retval = 'OK'
        else: 
            self.logger.error("Received unexpected data from client, disconnecting: %s"%str(new_bytes))
            retval = 'error'
            
        if retval != 'OK': self.disconnect(reconnect = False)
        return retval
        
    def select_error(self):
        self.logger.error("Select exceptional, disconnecting...")
        self.disconnect(reconnect = False)
    
    def can_write(self):
        try: bytes_sent = self.socket.send(self.bytes_to_write)
        except: 
            self.logger.error("Error sending bytes to socket")
            self.disconnect(reconnect = False)
        self.bytes_to_write = self.bytes_to_write[bytes_sent:]
        if self.state == 'sending_handshake' and self.bytes_to_write == b'': 
            self.state = 'waiting_init_packet'
        
    def __recv_collector(self, bytes):
        while len(bytes) > 0:
            if self.recevice_statemachine['bytes_left'] > len(bytes): n_bytes_read = len(bytes)
            else: n_bytes_read = self.recevice_statemachine['bytes_left']
            self.recevice_statemachine['recvd_bytes'] += bytes[:n_bytes_read]
            self.recevice_statemachine['bytes_left'] -= n_bytes_read
            bytes = bytes[n_bytes_read:]
            if self.recevice_statemachine['bytes_left'] == 0: 
                if self.__recv_stm_worker() != 'OK': break
        
    def can_read(self):
        try: bytes = self.socket.recv(4096)
        except: bytes = b''
        if bytes == b'': 
            self.logger.error("RECV returned b'', closing socket")
            self.disconnect(reconnect = True)
        else: self.__recv_collector(bytes)
        
    def disconnect(self, reconnect = False):
        if self.socket_opened:
            self.socket_opened = False
            self.socket.close()
            self.__send_state_update('disconnected')
            self.logger.warning("Connection closed")
        if reconnect:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setblocking(0)
            self.state = 'establishing_connection'
            self.__send_state_update('connecting')
        else: self.state = 'kill_me'
        

        
class NJSP_MULTISTREAMREADER(NBJSP):
    def __init__(self, logger = None, reader_name = None):
        super().__init__(logger = logger, name = reader_name)
        self.queue = queue.Queue(self.NJSP_PACKET_BUFFER_SIZE*5)
        self.abort_event = threading.Event()
        self.lock = threading.Lock()
        self.clients = dict()
        self.__gen_client_name = lambda ip, port: '%s:%d'%(ip,port)
        self.receiver_thread = threading.Thread(target=self.__receiver_thread_fxn, args=([]))
        self.receiver_thread.name = 'njsp_multi_rdr'
        self.receiver_thread.start()
        
    def kill(self):
        self.alive = False
        self.abort_event.set()
        self.receiver_thread.join()
        
    def add_client(self, ip, port, params = None):
        if ip == 'localhost' or ip == '': ip = '127.0.0.1'
        port = int(port)
        name = self.__gen_client_name(ip, port)
        client = NJSP_READER_HANDLER(self, name, (ip,port), params)
        with self.lock: self.clients.update({name:client})
        return name

    def remove_client(self, client_name):
        if client_name in self.clients: self.clients[client_name].disconnect(reconnect = False)
        
    def __receiver_thread_fxn(self):
        self.logger.debug("NJSP multistream reader thread started")
        self.alive = True
        while not self.abort_event.is_set(): 
            active_clients = dict()
            write_clients = dict()
            with self.lock:
                dead_clients = list()
                for client in self.clients.values():
                    if client.state == 'establishing_connection': client.try_to_connect()
                    if client.state == 'kill_me': dead_clients.append(client.name)
                    if client.socket_opened: 
                        active_clients.update({client.socket:client})
                        if len(client.bytes_to_write) > 0: 
                            write_clients.update({client.socket:client})
                for name in dead_clients:
                    del self.clients[name]
                            
            active_list = list(active_clients.keys())
            write_list = list(write_clients.keys())
            if len(active_list) > 0 or len(write_list) > 0:
                r, w, e = select.select(active_list, write_list, active_list, 1)
                #print(r, w, e)
                for s in e: active_clients[s].select_error()
                for s in w: write_clients[s].can_write()
                for s in r: active_clients[s].can_read()
            else:
                #print(active_list, write_list)
                time.sleep(1)
            
        for client in self.clients.values(): client.disconnect()

'''        
class NJSP_STREAMREADER_OVER_SSH(NJSP_STREAMREADER):
    def __init__(self, ssh_addr, ssh_login, ssh_pwd, remote_addr, **kwargs):
        from sshtunnel import SSHTunnelForwarder
        self.ssh_server = SSHTunnelForwarder(
            ssh_address_or_host = ssh_addr,
            ssh_username = ssh_login,
            ssh_password = ssh_pwd,
            remote_bind_address = remote_addr,
            compression = True
        )
        super().__init__(remote_addr, **kwargs)
        
    def _receiver_thread_fxn(self, remote_address):
        self.ssh_server.start()
        port = self.ssh_server.local_bind_port
        self.logger.info("Connecting over SSH tunnel on local port %d"%port)
        super()._receiver_thread_fxn(('localhost',port))
        self.ssh_server.stop()
        self.logger.info("SSH connection closed")
        
'''