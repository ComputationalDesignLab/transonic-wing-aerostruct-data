import torch
import numpy as np
import random

def set_seed(seed: int):
    """
        Helper function to set the seed for the PyTorch models

        Args
        ----
        seed: int
            random seed that must be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 


