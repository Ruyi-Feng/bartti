import torch
import os
from bartti.net import Bart
from train.params import params


def load(local_rank, pth):
    args = params()
    device = torch.device('npu', local_rank)
    model = Bart(args).float().to(device)
    if os.path.exists(pth):
        model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    return model

def save(model, path):
    torch.save(model.module.to('cpu').state_dict(), path)


if __name__ == '__main__':
    pth = './checkpoints/' + 'checkpoint_best.pth'
    path = './checkpoints/cpu/cpu_checkpoint_best.pth'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    model = load(local_rank, pth)
    save(model, path)

