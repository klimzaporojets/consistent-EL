import logging
import random

import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)
