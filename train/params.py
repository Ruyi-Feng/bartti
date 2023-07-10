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
    parser.add_argument('--train_epochs', type=int, default=128, help='total train epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default=200, help='all input seq will be compensated as max_seq_len')

    parser.add_argument('--c_in', type=int, default=4, help='output size')
    parser.add_argument('--c_out', type=int, default=6, help='output size')
    parser.add_argument('--frm_embed', type=int, default=16, help='output size')
    parser.add_argument('--id_embed', type=int, default=128, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--lradj', default='type1')
    args = parser.parse_args()

    return args

