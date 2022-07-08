import json

from detector.misc.globals import logger, ActionType

TRIGGER_PATH = '/var/lib/cloud9/ndas_rt/fifos/trigger'


def execute_action(action_id, action_on, inverse=False, action_message=None, action_address=None):
    action_id = min(action_id, ActionType.send_SMS.value)
    action_type = ActionType(action_id)
    inner_on = not action_on if inverse else action_on
    action_dic = {'type': str(action_type.name)}
    if action_type in [ActionType.relay_A, ActionType.relay_B]:
        action_dic['action'] = 'set' if inner_on else 'clear'
    if action_type == ActionType.send_SMS:
        if inner_on:
            action_dic['phone_number'] = action_address
            action_dic['text'] = action_message
        else:
            return
    actions_str = json.dumps({'actions': [action_dic]})
    logger.debug(f'actions_str:{actions_str}')
    try:
        with open(TRIGGER_PATH, 'w') as p:
            p.write(actions_str + '\n')
        logger.info("Trigger fired!")
    except Exception as ex:
        logger.error("Error writing data to trigger pipe:" + str(ex))
    # json_dic = {"actions": [{"type": "relay_A", "action": "set"},
    #                         {"type": "relay_B", "action": "clear"},
    #                         {"type": "send_SMS", "phone_number": "XXX", "text": "YYY"}]}


# print(execute_action(ActionType.send_SMS, 'Give me the pass, he-goat!', '7737'))
# print(execute_action(ActionType.relay_A, True))



