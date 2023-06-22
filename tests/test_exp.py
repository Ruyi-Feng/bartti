from train.exp import Exp_Main
from train.params import params

def test_exp():
    """
    args:
    --save_path
    --c_in
    --d_model
    --use_gpu
    --device_ids
    """
    args = params()
    bartti = Exp_Main(args)
    bartti.train()