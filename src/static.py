import os
import inspect
import logging


class CustomLogFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = os.getcwd() + os.sep

    def format(self, record):
        caller_file = inspect.stack()[9].filename.replace(self.root, '')
        caller_line = inspect.stack()[9].lineno
        clickable_link = f'{caller_file}:{caller_line}'
        record.clickable_link = clickable_link
        return super().format(record)


class Logger:
    _logger = None

    @staticmethod
    def get_logger():
        if Logger._logger is None:
            raise RuntimeError("Logger not initialized. Please call 'initialise_logging' first.")
        return Logger._logger

    @staticmethod
    def initialise(debug=False):
        level = logging.DEBUG if debug else logging.INFO

        formatter = CustomLogFormatter(
            '%(asctime)s - %(levelname)-5s - %(message)s (%(clickable_link)s)')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        Logger._logger = logging.getLogger("main")
        Logger._logger.setLevel(level)
        Logger._logger.handlers.clear()
        Logger._logger.addHandler(handler)

    @staticmethod
    def info(*args, **kwargs):
        return Logger._logger.info(*args, **kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        return Logger._logger.debug(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        return Logger._logger.warning(*args, **kwargs)
