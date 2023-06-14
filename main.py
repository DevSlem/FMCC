import argparse
import numpy as np
import random

from module.train import train
from module.test import test, eval

if __name__ == "__main__":
    # seed 고정
    random.seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true", help="Whether to run inference mode")
    parser.add_argument("-e", "--eval", action="store_true", help="Whether to run eval mode")
    
    args = parser.parse_args()
    
    if args.inference:
        print('=========test 시작=========')
        test()
    elif args.eval:
        print('=========test 시작=========')
        eval()
    else:
        print("=========train 시작=========")
        train()
