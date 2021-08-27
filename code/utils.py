import datetime
import logging
import time
import warnings


# ---------------------------------------------------------------------------


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, *exc):
        time_end = time.time()
        time_delta = datetime.timedelta(seconds=(time_end - self.time_start))
        if self.name:
            print(("Time for [{}]: {}".format(self.name, time_delta)))
        else:
            print(("Time: {}".format(time_delta)))


# ---------------------------------------------------------------------------


def configure_logging():
    warnings.filterwarnings("ignore")

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )


def init_random():
    import numpy as np
    import random

    np.random.seed(100)
    random.seed(100)

    try:
        import mxnet as mx

        mx.random.seed(10000)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
