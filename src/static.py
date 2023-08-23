import os
import inspect
import logging

TASKS = {
    'nlp': ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli'],
    'vis': ['cifar10', 'cifar100', 'pets', 'flowers', 'food', 'dtd', 'aircraft']
}

N_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'pets': 37,
    'flowers': 102,
    'food': 101,
    'dtd': 47,
    'aircraft': 100,
}


class CustomLogFormatter(logging.Formatter):
    """Custom log formatter with clickable links to caller file and line."""

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
    """Utility class for logging with custom formatting and logging levels."""

    _logger = None

    @staticmethod
    def get_logger():
        if Logger._logger is None:
            raise RuntimeError("Logger not initialized. Please call 'initialise_logging' first.")
        return Logger._logger

    @staticmethod
    def initialise(debug=False):
        """Initializes the logger with custom formatting and logging level."""
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


class GlobalState:
    """Container for global debug state."""
    debug = False