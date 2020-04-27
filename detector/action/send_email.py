from detector.filter_trigger.StaLtaTrigger import logger


def send_email(address, message):
    logger.info('send to address:' + address + '\nmessage:' + message)


