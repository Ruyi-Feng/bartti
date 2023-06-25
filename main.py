from train.exp import Exp_Main
from train.params import params
import os


if __name__ == '__main__':
    args = params()
    local_rank = int(os.environ['LOCAL_RANK'])
    bartti = Exp_Main(args, local_rank)
    bartti.train()
