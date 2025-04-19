import random
import numpy as np
import torch

def set_seed(seed):
    # CPU variables
    random.seed(seed)
    np.random.seed(seed)
    # Python
    torch.manual_seed(seed)
    # GPU variables
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False