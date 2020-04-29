from detector.misc.globals import logger


def send_email(address, message, on_off):
    if on_off:
        logger.info('send to address:' + address + '\nmessage:' + message)


