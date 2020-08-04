from detector.action.action_pipe import ActionType, execute_action
from detector.misc.globals import logger


def send_sms(address, message):
    logger.info('send to number:' + address + '\nmessage:' + message)
    execute_action(ActionType.send_SMS, message)


