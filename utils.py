# set_randomness
import random
import torch
import numpy as np

# log_print
from pathlib import Path
from absl import flags

FLAGS = flags.FLAGS


def set_randomness(seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return rng


def config_log_print(log_file_path):
    log_file_path = Path(log_file_path)
    assert not log_file_path.exists(), f'Log file {log_file_path} already exists.'
    assert log_file_path.parent.exists(), f'Invalid save directory {log_file_path.parent}.'

    def log_print(*args, **kwargs):
        with log_file_path.open('a') as f:
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)

    return log_print
