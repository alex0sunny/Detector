from detector.misc.globals import logger


def send_sms(address, message, on_off):
    if on_off:
        logger.info('send to number:' + address + '\nmessage:' + message)


