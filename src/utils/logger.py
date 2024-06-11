import logging
from tqdm import tqdm
import os
from datetime import datetime
import logging.config


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class Logger(logging.Logger):
    def __init__(
        self,
        name,
        CONFIG_DIR,
        LOG_DIR,
        level=logging.NOTSET,
    ):
        super().__init__(name, level)
        log_configs = {"dev": "logging.dev.ini", "prod": "logging.prod.ini"}
        config = log_configs.get(os.environ.get("ENV", "dev"), "logging.dev.ini")
        config_path = os.path.join(CONFIG_DIR, config)

        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")

        logging.config.fileConfig(
            config_path,
            disable_existing_loggers=False,
            defaults={"logfilename": os.path.join(LOG_DIR, f"{timestamp}.log")},
        )

        # Adding TqdmLoggingHandler to the logger
        self.addHandler(TqdmLoggingHandler())

    def info(self, msg, *args, **kwargs):
        tqdm.write(msg, end="")
        super().info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        super().debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        super().exception(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        super().log(level, msg, *args, **kwargs)

    def setLevel(self, level):
        super().setLevel(level)

    def isEnabledFor(self, level):
        return super().isEnabledFor(level)

    def addFilter(self, filter):
        super().addFilter(filter)

    def removeFilter(self, filter):
        super().removeFilter(filter)

    def addHandler(self, handler):
        super().addHandler(handler)

    def removeHandler(self, handler):
        super().removeHandler(handler)

    def clear(self):
        super().clear()
