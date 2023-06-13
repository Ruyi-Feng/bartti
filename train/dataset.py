from torch.utils.data import Dataset
from train.noise import derive


class Dataset_Bart(Dataset):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin"):
        self.train_idx = []
        self.dataset_length = 0
        self.idx_path = index_path
        self.data_path = data_path
        self.f_data = open(self.data_path, 'r')
        for line in open(self.idx_path, 'r'):
            self.train_idx.append([line[0], int(line[1]), int(line[2])])
            self.dataset_length += 1

    def _trans_to_array(self, info: str):
        """需要确定一下挖空和训练用的数据类型"""
        pass

    def __getitem__(self, index):
        head, tail = self.train_idx[index][1], self.train_idx[index][2]
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        x = self._trans_to_array(info)
        self.enc_x, self.dec_x, self.gt_x = derive(x)  # 这个是用来随机挖空的
        return self.enc_x, self.dec_x, self.gt_x

    def __len__(self):
        return self.dataset_length
