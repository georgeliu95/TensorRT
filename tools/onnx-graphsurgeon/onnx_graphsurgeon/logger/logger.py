import inspect
import enum
import time
import sys
import os


class LoggerIndent(object):
    def __init__(self, logger, indent):
        self.logger = logger
        self.old_indent = self.logger.logging_indent
        self.indent = indent

    def __enter__(self):
        self.logger.logging_indent = self.indent
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.logging_indent = self.old_indent


class LogMode(enum.IntEnum):
    EACH = 0 # Log the message each time
    ONCE = 1 # Log the message only once. The same message will not be logged again.


class Logger(object):
    ULTRA_VERBOSE = -10
    VERBOSE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def __init__(self, severity=INFO, colors=True):
        """
        Logger.

        Optional Args:
            severity (Logger.Severity): Messages below this severity are ignored.
        """
        self.severity = severity
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,  os.pardir))
        self.once_logged = set()
        self.colors = colors
        self.logging_indent = 0

    def indent(self, level=1):
        """
        Returns a context manager that indents all strings logged by the specified amount.
        """
        return LoggerIndent(self, level + self.logging_indent)

    @staticmethod
    def severity_logging_prefix(sev):
        default = "\033[1;"
        prefix = {
            Logger.ULTRA_VERBOSE: "90mUV",
            Logger.VERBOSE: "90mV",
            Logger.DEBUG: "90mD",
            Logger.INFO: "92mI",
            Logger.WARNING: "33mW",
            Logger.ERROR: "31mE",
            Logger.CRITICAL: "31mC",
        }[sev]
        return default + prefix

    def add_metadata(self, message, stack_depth):
        module = inspect.getmodule(sys._getframe(stack_depth))
        # Handle logging from the top-level of a module.
        if not module:
            module = inspect.getmodule(sys._getframe(stack_depth - 1))
        filename = module.__file__
        filename = os.path.relpath(filename, self.root_dir)
        # If the file is not located in trt_smeagol, use its basename instead.
        if os.pardir in filename:
            filename = os.path.basename(filename)

        message_lines = message.splitlines()
        message = "\n".join(["\t" * self.logging_indent + line for line in message_lines])
        return "({:}) [{:}:{:}] {:}".format(time.strftime("%H:%M:%S"), filename, sys._getframe(stack_depth).f_lineno, message)

    # If once is True, the logger will only log this message a single time. Useful in loops.
    # message may be a callable which returns a message. This way, only if the message needs to be logged is it ever generated.
    def log(self, message, severity, mode=LogMode.EACH):
        if severity < self.severity:
            return

        if callable(message):
            message = message()

        PREFIX_LEN = 12
        if mode == LogMode.ONCE:
            if message[PREFIX_LEN:] in self.once_logged:
                return
            self.once_logged.add(message[PREFIX_LEN:])

        print("{:} {:}\033[0m".format(Logger.severity_logging_prefix(severity), self.add_metadata(message, stack_depth=3)))

    def ultra_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ULTRA_VERBOSE, mode=mode)

    def verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.VERBOSE, mode=mode)

    def debug(self, message, mode=LogMode.EACH):
        self.log(message, Logger.DEBUG, mode=mode)

    def info(self, message, mode=LogMode.EACH):
        self.log(message, Logger.INFO, mode=mode)

    def warning(self, message, mode=LogMode.EACH):
        self.log(message, Logger.WARNING, mode=mode)

    def error(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ERROR, mode=mode)

    # Like error, but immediately exits.
    def critical(self, message):
        self.log(message, Logger.CRITICAL)
        raise Exception("Error encountered - see logging output for details") from None # Erase exception chain

global G_LOGGER
G_LOGGER = Logger()
