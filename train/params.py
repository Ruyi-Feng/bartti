import argparse
import torch

def params():
    parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--index_path', type=str, default='./data/index.bin')
    parser.add_argument('--data_path', type=str, default='./data/data.bin')
    parser.add_argument('--interval', type=float, default=0.03)

    parser.add_argument('--is_train', type=bool, default=True, help='if True is train model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='original leaning rate')
    parser.add_argument('--train_epochs', type=int, default=3, help='total train epoch')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default=400, help='all input seq will be compensated as max_seq_len')

    parser.add_argument('--c_in', type=int, default=5, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default='0', help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lradj', default='type1')
    args = parser.parse_args()
    args.use_gpu = (torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    return args

