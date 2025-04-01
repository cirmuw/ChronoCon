import logging
import datetime
import sys


# setup
SCRATCH_LOG_DIR = "output/scratch_output/scratch_log/"


class Logger:
    # nr_of_main_loggers = 0

    root_level = logging.DEBUG
    root_file_level = logging.DEBUG
    root_stream_level = logging.WARNING
    root_file = SCRATCH_LOG_DIR + 'main_' + \
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
    root_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    root_mode = 'a'
    root_stream = sys.stdout

    @staticmethod
    def root_logger(level=root_level,
                    handler_format=root_format,
                    file_level=root_file_level,
                    file_handler_path=root_file,
                    write_mode=root_mode,
                    stream_handler_type=root_stream,
                    stream_level=root_stream_level,
                    clear_existing_handlers=True,
                    add_to_existing_handlers=False):

        logger = logging.getLogger()

        if clear_existing_handlers and len(logger.handlers):
            logger.handlers.clear()

        if not len(logger.handlers) or add_to_existing_handlers:
            # set logger level
            logger.setLevel(level)

            # create formatter and add it to the handlers
            formatter = logging.Formatter(handler_format)

            # create file handler which logs even debug messages
            fh = logging.FileHandler(file_handler_path, mode=write_mode)
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # create console handler with a higher log level
            ch = logging.StreamHandler(stream_handler_type)
            ch.setLevel(stream_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    @staticmethod
    def sub_logger(name, level=root_level,
                   handler_format=root_format,
                   file_handler_path=None,
                   file_level=None,
                   write_mode=None,
                   stream_handler_type=None,
                   stream_level=None,
                   clear_existing_handlers=None,
                   add_to_existing_handlers=False
                   ):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if clear_existing_handlers is not None and clear_existing_handlers and len(logger.handlers):
            logger.handlers.clear()

        if not len(logger.handlers) or add_to_existing_handlers:
            if handler_format is not None:
                # create formatter and add it to the handlers
                formatter = logging.Formatter(handler_format)
            if stream_handler_type is not None:
                # create stream handler
                ch = logging.StreamHandler(stream_handler_type)
                if stream_level is not None:
                    ch.setLevel(file_level)
                if handler_format is not None:
                    ch.setFormatter(formatter)
                logger.addHandler(ch)
            if file_handler_path is not None:
                # create stream handler which logs even debug messages
                fh = logging.FileHandler(file_handler_path)
                if write_mode is not None:
                    fh.mode = write_mode
                if file_level is not None:
                    fh.setLevel(file_level)
                if handler_format is not None:
                    fh.setFormatter(formatter)
                logger.addHandler(fh)

        return logger


if __name__ == '__main__':

    # Logger.root_file = ...
    # Logger.file_level = ...
    log = Logger.root_logger()
    import utils.utils_archive.scratch_log_4 as auxiliary_module

    log.info('creating an instance of auxiliary_module.Auxiliary')
    a = auxiliary_module.Auxiliary()
    log.info('created an instance of auxiliary_module.Auxiliary')

    log.info('calling auxiliary_module.Auxiliary.do_something')
    a.do_something()
    log.info('finished auxiliary_module.Auxiliary.do_something')

    a.do_error()

    log.info('calling auxiliary_module.some_function()')
    auxiliary_module.some_function()
    log.info('done with auxiliary_module.some_function()')

    log.debug('debug message')
    log.info('info message')
    log.warning('warn message')
    log.error('error message')
    log.critical('critical message')
    log.debug("this was written in main")

    """
    # remember to close the handlers
    for handler in log.handlers:
        handler.close()
        log.removeFilter(handler)
    """
#
