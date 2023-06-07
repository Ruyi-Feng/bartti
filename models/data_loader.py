from torch.utils.data import Dataset


class Dataset_ETT_hour(Dataset):
    def __init__(self):
        pass
        self.__read_data__()

    def __read_data__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1