import sys
from utils.logging_class import Logger
import logging

scratch_log_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/scratch_output/scratch_log/"

# create logger
module_logger = Logger.sub_logger('spam_application.auxiliary')


class Auxiliary:
    def __init__(self):
        self.logger = Logger.sub_logger(module_logger.name + '.' + type(self).__name__)
        self.logger.info('creating an instance of Auxiliary')

    def do_something(self):
        self.logger.info('doing something')
        a = 1 + 1
        self.logger.info('done doing something')

    def do_error(self):
        self.logger.error('auxiliary error')


def some_function():
    module_logger.info('received a call to "some_function"')


def logloglog():
    logging.debug("this was written in 2")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG, filemode='a')

    logloglog()

#
