import numpy as np
import torch
from datetime import datetime

def now():
    return datetime.now().strftime("%H:%M:%S")

def set_seed(seed: int):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
