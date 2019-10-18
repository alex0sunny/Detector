import zmq


DEFAULT_PAGE = '\r\n'.join([
    "HTTP/1.0 200 OK",
    "Content-Type: text/plain",
    "",
    "Hello, World!",
])


context = zmq.Context()

router = context.socket(zmq.ROUTER)
router.router_raw = True
router.bind('tcp://*:8080')

while True:
    identity = router.recv()
    request = router.recv()

    # send Hello World page
    router.send(identity, zmq.SNDMORE)
    router.send(DEFAULT_PAGE.encode())

    # Close connection to browser
    router.send(identity, zmq.SNDMORE)
    router.send(b'')

