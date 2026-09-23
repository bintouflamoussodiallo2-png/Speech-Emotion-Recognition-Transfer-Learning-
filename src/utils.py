import os
import random 
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Ensure full reproducibility across runs.
    Call this at the very beginning of train.py before any model or data initialization.

    Args:
        seed : integer seed (default 42 — conventional in ML research)

    Why each line matters:
        - random       : Python's built-in random module
        - numpy        : used by librosa and dataset augmentations
        - torch        : CPU operations
        - torch.cuda   : GPU operations
        - PYTHONHASHSEED: controls Python's hash randomization
        - cudnn        : ensures deterministic convolution algorithms
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # for a determinsitic algo on GPU with garanted reproductibility, we use cddnn
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False # slower but guaranted reproductibility
    

