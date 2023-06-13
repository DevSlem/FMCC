import numpy as np
import random

from module.train import train
from module.test import test

if __name__ == "__main__":
    # seed 고정
    random.seed(0)
    np.random.seed(0)
    
    print('=========train 시작=========')
    train()
    
    print()
    
    print('=========test 시작=========')
    test()