from time import sleep

import zmq

from detector.filter_trigger.StaLtaTrigger import logger
from detector.misc.globals import Port, Subscription


def action_process(action_id, send_func, rules=[], args={}, pet=0, infinite=False):
    #logger.debug('start process for action ' + str(action_id))
    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect('tcp://localhost:' + str(Port.proxy.value))
    action_id_s = '%02d' % action_id
    socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.test.value + action_id_s.encode())
    if action_id in [1, 2]:
        args['relay_n'] = action_id
        args['on_off'] = 0
        send_func(**args)  # initialization
        #logger.debug('use test subscription:' + str(Subscription.test.value + action_id_s.encode()))
    for rule_id in rules:
        rule_id_s = '%02d' % rule_id
        socket_sub.setsockopt(zmq.SUBSCRIBE, Subscription.rule.value + rule_id_s.encode())
    trigger = 0
    wait_pet = False
    infinite_on = False
    while True:
        # if action_id in [1, 2]:
        #     logger.debug('wait event')
        if socket_sub.poll(max(pet, 1) * 1000):     # if pet=0 wait one second second
            logger.debug('trigger:' + str(trigger))
            bin_message = socket_sub.recv()
            if bin_message[:1] == Subscription.test.value:
                if infinite:
                    infinite_on = False
                    trigger = 0
                if trigger == 0:
                    if action_id in [1, 2]:
                        logger.debug('test event')
                    args['on_off'] = 1
                    send_func(**args)
                    wait_pet = True
                    # sleep(max(pet, 1))
                    # args['on_off'] = 0
                    # send_func(**args)
            elif not infinite_on:
                if int(bin_message[3:4]):
                    trigger_val = 1
                    if infinite:
                        infinite_on = True
                else:
                    trigger_val = -1
                # if action_id in [1, 2]:
                #     if trigger_val > 0:
                #         logger.debug('trigger event')
                #     else:
                #         logger.debug('detrigger event')
                trigger += trigger_val
                if trigger == 1 and trigger_val == 1:
                    args['on_off'] = 1
                    send_func(**args)
                if trigger == 0:
                    if pet:
                        wait_pet = True
                    else:
                        args['on_off'] = 0
                        send_func(**args)
                if trigger < 0:
                    if action_id in [1, 2]:
                        logger.warning('unexpected detriggering')
                    trigger = 0
        else:
            # if action_id in [1, 2]:
            #     logger.debug('no event, pet:' + str(pet))
            if trigger == 0 and wait_pet and not infinite_on:
                wait_pet = False
                args['on_off'] = 0
                send_func(**args)

