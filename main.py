from train.exp import Exp_Main
from train.params import params


if __name__ == '__main__':
    args = params()
    bartti = Exp_Main(args)
    bartti.train()
