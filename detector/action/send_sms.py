from detector.filter_trigger.StaLtaTrigger import logger


def send_sms(address, message):
    logger.info('send to number:' + address + '\nmessage:' + message)


