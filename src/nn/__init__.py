import logging
import os
import platform
import sys

LOGGING_NAME = "v4"
# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
VERBOSE = str(os.getenv("V4_VERBOSE", True)).lower() == "true"
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None
LINUX, WINDOWS = (platform.system() == x for x in ["Linux", "Windows"])


def set_logging(name="LOGGING_NAME", verbose=True):
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                return emojis(super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string
