import json
from enum import Enum

from detector.filter_trigger.StaLtaTrigger import logger


class ActionType(Enum):
    relay_A = 1
    relay_B = 2
    send_SIGNAL = 3
    send_SMS = 4


def execute_action(action_type, action_message, action_address=None):
    action_dic = {'type': str(action_type.name)}
    if action_type in [ActionType.relay_A, ActionType.relay_B]:
        action = 'clear'
        if action_message:
            action = 'set'
        action_dic['action'] = action
    if action_type == ActionType.send_SMS:
        action_dic['phone_number'] = action_address
        action_dic['text'] = action_message
    actions_str = json.dumps({"actions": [action_dic]})
    try:
        with open('/var/lib/cloud9/ndas_rt/fifos/trigger', 'w') as p:
            p.write(actions_str)
        logger.info("Trigger fired!")
    except Exception as ex:
        logger.error("Error writing data to trigger pipe:" + str(ex))
    return actions_str
    # json_dic = {"actions": [{"type": "relay_A", "action": "set"},
    #                         {"type": "relay_B", "action": "clear"},
    #                         {"type": "send_SMS", "phone_number": "XXX", "text": "YYY"}]}


# print(execute_action(ActionType.send_SMS, 'Give me the pass, he-goat!', '7737'))
# print(execute_action(ActionType.relay_A, True))



