import zmq


DEFAULT_PAGE = '\r\n'.join([
    "HTTP/1.0 200 OK",
    "Content-Type: text/plain",
    "",
    "Hello, World!",
])


context = zmq.Context()

router = context.socket(zmq.STREAM)
router.bind('tcp://*:8080')

while True:
    identity = router.recv()
    router.recv()
    router.recv()
    request = router.recv()
    router.send(identity, zmq.SNDMORE)
    router.send(DEFAULT_PAGE.encode())
    router.send(identity, zmq.SNDMORE)
    router.send(b'')

