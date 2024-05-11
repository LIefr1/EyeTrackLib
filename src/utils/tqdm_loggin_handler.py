import logging
from tqdm import tqdm


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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Add TqdmLoggingHandler to the logger
logger.addHandler(TqdmLoggingHandler())

# Example usage
logger.info("This message will be shown in tqdm progress bar.")
