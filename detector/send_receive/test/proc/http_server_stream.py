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
    print('receive identity')
    empty = router.recv()
    if empty:
        print('empty string expected:' + empty)
    else:
        print('received empty')
    if not identity == router.recv():
        print('no identity')
    else:
        print('received identity')
    request = router.recv()
    print('received request:' + str(request))

    # send Hello World page
    router.send(identity, zmq.SNDMORE)
    print('identity sent')
    router.send(DEFAULT_PAGE.encode())
    print('response sent')

    # Close connection to browser
    router.send(identity, zmq.SNDMORE)
    print('identity sent')
    router.send(b'')
    print('empty sent')

