from torch.utils.data import Dataset
from utils.noise import derive


class Dataset_Bart(Dataset):
    def __init__(self):
        pass
        self.__read_data__()

    def __read_data__(self):
        pass
        self.enc_x, self.dec_x, self.gt_x = derive(x)
        # generate random mask here

    def __getitem__(self, index):
        pass
    return enc_x, dec_x, gt_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1