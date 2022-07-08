import json, base64, bson, threading, collections, queue, logging, asyncio, signal, socket, platform

if platform.system() == 'Windows': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class NJSP_LOGGER_ADAPTER(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['prefix'], msg), kwargs


class NJSP_HANDLE_BASE:
    def __init__(self, njsp, tp, ip, port, prefix):
        self.njsp, self.tp, self.ip, self.port, self.prefix = njsp, tp, ip, int(port), prefix
        if prefix != None:
            self.name = '%s:%s:%s:%d' % (tp, prefix, ip, int(port))
        else:
            self.name = '%s:%s:%d' % (tp, ip, int(port))
        self.logger = NJSP_LOGGER_ADAPTER(njsp.logger, {'prefix': self.name})
        self.read_packet = self.read_packet_json
        self.encode_hdr_and_payload = self.encode_hdr_and_payload_json
        self.split_to_datatypes_and_encode = self.split_to_datatypes_and_encode_to_json
        self.PROTOCOL_ID = self.njsp.ID_JSON
        self.bson = False

    def _to_bson_mode(self):
        self.read_packet = self.read_packet_bson
        self.encode_hdr_and_payload = self.encode_hdr_and_payload_bson
        self.split_to_datatypes_and_encode = self.split_to_datatypes_and_encode_to_bson
        self.PROTOCOL_ID = self.njsp.ID_BSON
        self.bson = True
        self.logger.debug("Using BSON mode")

    async def init_async(self):
        try:
            self.closed = asyncio.Event()
            async with self.njsp.aio_lock:
                self.njsp.handles.update({self.name: self})
            self.future = asyncio.create_task(self.task())
            res = 'OK'
        except Exception as e:
            self.logger.error(repr(e))
            res = 'error'
        finally:
            return res

    async def read_packet_json(self, reader):
        hdr = await reader.readexactly(8)
        length = int(hdr.decode('ASCII'), 16)
        payload = await reader.readexactly(length)
        retval = json.loads(payload.decode('ASCII'))
        for stream in retval.get('streams', dict()).values():
            for ch_name, ch_signal in stream.get('samples', dict()).items():
                data = base64.decodebytes(ch_signal.encode('ASCII'))
                stream['samples'].update({ch_name: data})
        return retval

    async def read_packet_bson(self, reader):
        hdr = await reader.readexactly(4)
        length = int.from_bytes(hdr, byteorder='big', signed=False)
        payload = await reader.readexactly(length)
        return bson.loads(payload)

    def encode_hdr_and_payload_json(self, dict_obj):
        for stream in dict_obj.get('streams', dict()).values():
            for ch_name, ch_signal in stream.get('samples', dict()).items():
                encoded_signal = base64.encodebytes(ch_signal).decode('ASCII')
                stream['samples'].update({ch_name: encoded_signal})
        payload = json.dumps(dict_obj).encode('ASCII')
        hdr = format(len(payload), '08x').encode('ASCII')
        return hdr + payload

    def encode_hdr_and_payload_bson(self, dict_obj):
        payload = bson.dumps(dict_obj)
        hdr = int.to_bytes(len(payload), length=4, byteorder='big')
        return hdr + payload

    def split_to_datatypes_and_encode_to_json(self, dict_obj):
        out_packet = dict()
        for data_type, payload in dict_obj.items():
            out_packet.update({
                data_type: self.encode_hdr_and_payload_json({data_type: payload})
            })
        return out_packet

    def split_to_datatypes_and_encode_to_bson(self, dict_obj):
        out_packet = dict()
        for data_type, payload in dict_obj.items():
            out_packet.update({
                data_type: self.encode_hdr_and_payload_bson({data_type: payload})
            })
        return out_packet

    async def task(self):
        try:
            await self._task_loop()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error('Main loop exception:\n\t%s' % repr(e))
        finally:
            self.logger.debug('Task stopped')
            self.closed.set()
            async with self.njsp.aio_lock:
                del self.njsp.handles[self.name]


class NJSP_LOCAL_CLIENT(NJSP_HANDLE_BASE):
    def __init__(self, reader, streamer):
        self.reader = reader
        self.reader_name = reader.name
        self.subscriptions = reader.handshake['subscriptions']
        self.init_packet = streamer.params['init_packet']
        self.queue = reader.queue
        self.streamer = streamer
        super().__init__(reader.njsp, NJSP.LOCAL_CLIENT, '', streamer.port, reader.prefix)

    def write_nonblock(self, packet):
        for datatype, data in packet.items():
            if datatype in self.subscriptions:
                try:
                    self.queue.put_nowait({self.reader_name: {datatype: data}})
                except queue.Full:
                    self.logger.error("Queue full, closing connection")
                    self.closed.set()
            if datatype == 'abort': self.closed.set()

    async def _task_loop(self):
        self.logger.info("Shortcut found, connecting to %s directly" % (self.streamer.name))
        self.queue.put_nowait({self.reader_name: {'connection_state': 'connected'}})
        self.queue.put_nowait({self.reader_name: self.init_packet})
        self.logger.debug('Subscriptions: %s' % str(self.subscriptions))
        if self.reader.handshake.get('flush_buffer', False):
            for packet in list(self.streamer.ringbuffer): self.write_nonblock(packet)
        self.streamer.register_client(self)
        self.logger.info('Connected')
        awaitables = [
            asyncio.create_task(self.closed.wait()),
            asyncio.create_task(self.reader.closed.wait()),
            asyncio.create_task(self.streamer.closed.wait())
        ]
        done, pending = await asyncio.wait(awaitables, return_when=asyncio.FIRST_COMPLETED)
        for fut in pending: fut.cancel


class NJSP_READER(NJSP_HANDLE_BASE):
    def __init__(self, njsp, ip, port, prefix, params, rd_queue):
        if ip == 'localhost' or ip == '': ip = '127.0.0.1'
        self.params = params
        self.handshake = params['handshake']
        self.queue = rd_queue
        super().__init__(njsp, NJSP.READER, ip, port, prefix)
        if params.get('bson', False): self._to_bson_mode()
        self.logger.info("Reader created")

    async def find_local_streamer(self):
        if self.ip != '127.0.0.1': return False
        local_streamer = None
        async with self.njsp.aio_lock:
            for streamer in self.njsp.handles.values():
                if streamer.tp == NJSP.STREAMER and streamer.port == self.port:
                    return streamer

    async def _internal_loop(self):
        reader, writer = await asyncio.open_connection(self.ip, self.port)
        self.logger.debug('Connection established')
        handshake_packet = self.encode_hdr_and_payload({'handshake': self.handshake})
        writer.write(self.PROTOCOL_ID + handshake_packet)
        await writer.drain()
        idbytes = await reader.readexactly(len(self.PROTOCOL_ID))
        if idbytes != self.PROTOCOL_ID:
            raise RuntimeError('Incorrect protocol ID: %s' % str(idbytes))
        init_packet = await self.read_packet(reader)
        self.queue.put_nowait({self.name: {'connection_state': 'connected'}})
        self.queue.put_nowait({self.name: init_packet})
        self.last_state = "Connected"
        self.logger.info("Connected")
        while True:
            packet = await self.read_packet(reader)
            self.queue.put_nowait({self.name: packet})
            if 'abort' in packet: raise RuntimeError('Connection aborted by streamer')

    async def _task_loop(self):
        self.last_state = "Disconnected"
        while True:
            if self.last_state == "Disconnected":
                self.queue.put_nowait({self.name: {'connection_state': 'connecting'}})
                self.last_state = 'Connecting'
            local_streamer = await self.find_local_streamer()

            try:
                if local_streamer is None or local_streamer is False:
                    await self._internal_loop()
                else:
                    client = NJSP_LOCAL_CLIENT(self, local_streamer)
                    await client.init_async()
                    self.last_state = "Connected"
                    await client.closed.wait()
            except queue.Full:
                self.logger.error("Queue full")
                break
            except ConnectionError as e:
                self.logger.debug('Connection error:\n\t%s' % repr(e))
            except Exception as e:
                self.logger.error('Reader loop exception:\n\t%s' % repr(e))
                break

            if self.last_state == "Connected":
                self.queue.put_nowait({self.name: {'connection_state': 'disconnected'}})
                self.last_state = "Disconnected"

            if not self.params.get('reconnect', False): break
            await asyncio.sleep(self.params.get('reconnect_period', 30))

    @staticmethod
    def check_params(dict_obj):
        if 'handshake' not in dict_obj:
            dict_obj.update({'handshake': dict()})
        dict_obj['handshake'].update({'protocol_version': NJSP.PROTOCOL_VERIOSN})
        if 'subscriptions' not in dict_obj['handshake']:
            dict_obj['handshake'].update({'subscriptions': ['status', 'log', 'streams', 'alarms']})
        if 'client_name' not in dict_obj['handshake']:
            dict_obj['handshake'].update({'client_name': socket.gethostname()})
        if 'flush_buffer' not in dict_obj['handshake']:
            dict_obj['handshake'].update({'flush_buffer': False})

    @staticmethod
    def construct_name(ip, port, prefix):
        if ip == 'localhost' or ip == '': ip = '127.0.0.1'
        if prefix != None:
            return 'RD:%s:%s:%d' % (prefix, ip, int(port))
        else:
            return 'RD:%s:%d' % (ip, int(port))


class NJSP_STREAMER_CLIENT(NJSP_HANDLE_BASE):
    def __init__(self, streamer, reader, writer):
        self.streamer, self.reader, self.writer = streamer, reader, writer
        print(f'params:{writer.get_extra_info("peername")}')
        super().__init__(streamer.njsp, NJSP.CLIENT, *writer.get_extra_info('peername'), None)

    def write_nonblock(self, encoded_packet):
        try:
            for datatype, datapacket in encoded_packet.items():
                if datatype in self.subscriptions:
                    try:
                        self.writer.write(datapacket)
                    except asyncio.LimitOverrunError:
                        self.writer.close()
                        self.logger.warning('Buffer overflow, breaking connection')
        except Exception as e:
            self.writer.close()
            self.logger.error('Write exception\n\t%s', repr(e))

    async def _task_loop(self):
        self.logger.debug('Connection from %s:%d, handshaking...' % (self.ip, self.port))
        idbytes = await self.reader.readexactly(len(self.PROTOCOL_ID))
        if idbytes != self.njsp.ID_JSON:
            if idbytes == self.njsp.ID_BSON:
                self._to_bson_mode()
            else:
                raise RuntimeError('Incorrect protocol ID: %s' % str(idbytes))
        packet = await self.read_packet(self.reader)
        if 'handshake' not in packet:
            raise RuntimeError('Handshake packet expected!')
        handshake = packet['handshake']
        ver = handshake.get('protocol_version', 'unknown')
        if ver != NJSP.PROTOCOL_VERIOSN:
            raise RuntimeError('Incompatible protocol ver!')
        if 'client_name' in handshake:
            self.logger.info("Client name is %s" % handshake['client_name'])
        self.subscriptions = handshake.get('subscriptions', dict())
        self.logger.debug('Subscriptions: %s' % str(self.subscriptions))
        init_packet = self.encode_hdr_and_payload(self.streamer.params['init_packet'])
        self.writer.write(self.PROTOCOL_ID + init_packet)
        if handshake.get('flush_buffer', False):
            async with self.streamer.ringbuffer_lock:
                self.logger.debug('Flushing buffer...')
                for packet in list(self.streamer.ringbuffer):
                    encoded_packet = self.split_to_datatypes_and_encode(packet)
                    self.write_nonblock(encoded_packet)
                    await asyncio.wait_for(self.writer.drain(), timeout=0.25)
        self.streamer.register_client(self)
        self.logger.info('Connected')
        try:
            await self.writer.wait_closed()
        except BrokenPipeError:
            self.logger.info("Socket closed: BrokenPipeError")


class NJSP_STREAMER(NJSP_HANDLE_BASE):
    def __init__(self, njsp, ip, port, params):
        if ip == 'localhost': ip = ''
        self.params = params
        self.clients = dict()
        self.local_clients = dict()
        self.bson_clients = dict()
        super().__init__(njsp, NJSP.STREAMER, ip, port, None)
        self.logger.info('Streamer created')

    async def init_async(self):
        self.ringbuffer = collections.deque(list(), self.params.get('ringbuffer_size', 10))
        self.ringbuffer_lock = asyncio.Lock()
        return await super().init_async()

    def register_client(self, client_handle):
        if client_handle.tp == NJSP.CLIENT:
            if client_handle.bson:
                self.bson_clients.update({client_handle.name: client_handle})
            else:
                self.clients.update({client_handle.name: client_handle})
        else:
            self.local_clients.update({client_handle.name: client_handle})
        self.logger.debug("Client %s registered" % client_handle.name)

    async def remove_dead_clients(self):
        while not self.closed.is_set():
            dead_clients, dead_local_clients, dead_bson_clients = list(), list(), list()
            for name, handle in self.clients.items():
                if handle.closed.is_set(): dead_clients.append(name)
            for name, handle in self.bson_clients.items():
                if handle.closed.is_set(): dead_bson_clients.append(name)
            for name, handle in self.local_clients.items():
                if handle.closed.is_set(): dead_local_clients.append(name)
            for name in dead_clients: del self.clients[name]
            for name in dead_bson_clients: del self.bson_clients[name]
            for name in dead_local_clients: del self.local_clients[name]
            total = dead_clients + dead_local_clients + dead_bson_clients
            if len(total) > 0: self.logger.info("Removed clients: %s" % str(total))
            try:
                await asyncio.wait_for(self.closed.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass

    async def broadcast(self, packet):
        try:
            async with self.ringbuffer_lock:
                self.ringbuffer.append(packet)
                if len(self.clients) > 0:
                    encoded_packet = self.split_to_datatypes_and_encode_to_json(packet)
                    for client in self.clients.values():
                        if not client.closed.is_set():
                            client.write_nonblock(encoded_packet)
                if len(self.bson_clients) > 0:
                    encoded_packet = self.split_to_datatypes_and_encode_to_bson(packet)
                    for client in self.bson_clients.values():
                        if not client.closed.is_set():
                            client.write_nonblock(encoded_packet)
                for client_name, client in self.local_clients.items():
                    client.write_nonblock(packet)
        except Exception as e:
            self.logger.error("Broadcast exception:\n\t%s" % repr(e))

    async def _task_loop(self):
        handler = lambda r, w: NJSP_STREAMER_CLIENT(self, r, w).init_async()
        server = await asyncio.start_server(handler, self.ip, self.port, reuse_address=True)
        asyncio.create_task(self.remove_dead_clients())
        async with server: await server.serve_forever()

    @staticmethod
    def construct_name(ip, port):
        return '%s:%s:%d' % (NJSP.STREAMER, ('' if ip == 'localhost' else ip), int(port))

    @staticmethod
    def check_params(dict_obj):
        if 'init_packet' not in dict_obj: raise RuntimeError('Init packet is missing!')
        if 'parameters' not in dict_obj['init_packet']: raise RuntimeError('Incorrect init packet!')
        params = dict_obj['init_packet']['parameters']
        ver = params.get('parameters', NJSP.PROTOCOL_VERIOSN)
        if ver != NJSP.PROTOCOL_VERIOSN:  raise RuntimeError('Incorrect protocol version!')
        for stream in params.get('streams', dict()).values():
            if 'data_format' not in stream: raise RuntimeError('Data format is missing!')
            if 'sample_rate' not in stream: raise RuntimeError('Stream sample rate is missing!')
            if 'channels' not in stream: raise RuntimeError('Channels params are missing!')


class NJSP:
    PROTOCOL_VERIOSN = 3.0
    ID_JSON = b'NJSP\0\0'
    ID_BSON = b'NBJSP\0'
    READER = 'RD'
    STREAMER = 'ST'
    CLIENT = 'CL'
    LOCAL_CLIENT = 'LO'

    def __init__(self, logger=logging.getLogger(), log_level=logging.INFO):
        #self.logger = NJSP_LOGGER_ADAPTER(logger, {'prefix': 'NJSP'})
        self.logger = logger
        self.logger.setLevel(log_level)
        self.handles = dict()
        start_event = threading.Event()
        self.thread_lock = threading.Lock()
        self.thread = threading.Thread(target=asyncio.run, args=([self.main(start_event)]))
        self.thread.start()
        start_event.wait()

    def add_streamer(self, ip, port, params):
        name = NJSP_STREAMER.construct_name(ip, int(port))
        NJSP_STREAMER.check_params(params)
        with self.thread_lock:
            if name in self.handles: raise ValueError('Streamer %s already exists' % name)
            streamer = NJSP_STREAMER(self, ip, int(port), params)
            fut = asyncio.run_coroutine_threadsafe(streamer.init_async(), self.ioloop)
            fut.result()
        return name

    def add_reader(self, ip, port, prefix, params, rd_queue):
        name = NJSP_READER.construct_name(ip, int(port), prefix)
        NJSP_READER.check_params(params)
        with self.thread_lock:
            if name in self.handles: raise ValueError('Reader %s already exists' % name)
            reader = NJSP_READER(self, ip, int(port), prefix, params, rd_queue)
            fut = asyncio.run_coroutine_threadsafe(reader.init_async(), self.ioloop)
            fut.result()
        return name

    def is_alive(self, name):
        if name == None:  return False
        if name not in self.handles: return False
        try:
            return not self.handles[name].closed.is_set()
        except Exception as e:
            self.logger.warning(repr(e))
            return False

    def connected_clients(self, name):
        if name == None: return 0
        try:
            h = self.handles[name]
            return len(h.clients) + len(h.local_clients) + len(h.bson_clients)
        except Exception as e:
            self.logger.warning(repr(e))
            return 0

    def remove(self, name):
        if name == None: return
        try:
            self.ioloop.call_soon_threadsafe(self.handles[name].future.cancel)
            self.logger.info('Handle removed: %s' % name)
        except Exception as e:
            self.logger.warning(repr(e))

    def broadcast_data(self, name, packet):
        try:
            asyncio.run_coroutine_threadsafe(self.handles[name].broadcast(packet), self.ioloop)
        except Exception as e:
            self.logger.warning(repr(e))

    async def main(self, start_event):
        self.logger.debug('Asyncio loop started')
        self.abort_event = asyncio.Event()
        self.aio_lock = asyncio.Lock()
        self.ioloop = asyncio.get_event_loop()
        start_event.set()
        while not self.abort_event.is_set():
            stats = {NJSP.READER: 0, NJSP.STREAMER: 0, NJSP.CLIENT: 0, NJSP.LOCAL_CLIENT: 0}
            async with self.aio_lock:
                for h in self.handles.values(): stats.update({h.tp: stats[h.tp] + 1})
            self.logger.debug(stats)
            try:
                await asyncio.wait_for(self.abort_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
        self.logger.debug('Asyncio loop finished')

    def kill(self):
        self.ioloop.call_soon_threadsafe(self.abort_event.set())
        self.thread.join()
        self.logger.debug('NJSP killed')