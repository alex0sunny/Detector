from detector.action.action_pipe import ActionType, execute_action
from detector.misc.globals import logger


def send_sms(address, message, on_off):
    if on_off:
        logger.info('send to number:' + address + '\nmessage:' + message)
        execute_action(ActionType.send_SMS, on_off)


