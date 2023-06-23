from train.exp import Exp_Main
from train.params import params

def test_exp():
    args = params()
    bartti = Exp_Main(args)
    bartti.train()

if __name__ == '__main__':
    test_exp()
