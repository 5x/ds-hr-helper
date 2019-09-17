import logging

logger = logging.getLogger()
logger_handler = logging.StreamHandler()
log_template = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
log_dt_template = '%d/%m/%Y %H:%M:%S'
formatter = logging.Formatter(log_template, log_dt_template)
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)
logger.setLevel(logging.DEBUG)


def http_logger(url, state):
    request_num = len(state.data)
    links_count = len(state.fetched_urls)
    logger.info('GET[%s, %s]: %s', request_num, links_count, url)
