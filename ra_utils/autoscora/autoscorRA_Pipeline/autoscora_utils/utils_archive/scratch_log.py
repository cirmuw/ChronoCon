import logging
import sys
from utils_2.scratch_log_2 import logloglog
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
scratch_log_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/scratch_output/scratch_log/"
# logging.basicConfig(filename=scratch_log_dir + "log_main",  # stream=sys.stdout,
#                     level=logging.DEBUG, filemode='a')

logger = logging.getLogger('main_logger_3')
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler(scratch_log_dir + 'main_' + current_time + '.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
logger.debug("this was written in main")

logger.debug("blu")
logloglog()

#
