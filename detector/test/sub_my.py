import zmq

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.STREAM)

socket.bind("tcp://*:5556")

# Subscribe to zipcode, default is NYC, 10001
#topicfilter = "10001"
#socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

total_value = 0
while True:
    id = socket.recv()
    print('id:' + str(id))
    str_data = str(socket.recv())
    print('recv:' + str_data)
    id = socket.recv()   # id
    print('id:' + str(id))
    str_data = str(socket.recv())
    print('recv:' + str_data)
