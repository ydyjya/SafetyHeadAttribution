from datetime import datetime
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    return formatted_time
