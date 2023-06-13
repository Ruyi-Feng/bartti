from torch.utils.data import Dataset
from train.noise import derive


class Dataset_Bart(Dataset):
    def __init__(self):
        """
        把索引读进内存里，初始化长度存储。
        """
        self.dataset_lenghth

    def __getitem__(self, index):
        """
        根据索引文件查询实际文件中内容。
        通过索引找文件存储在哪个文件里，再读相应的字节部分。

        得到索引之后f.seek() f.read()

        """

        self.enc_x, self.dec_x, self.gt_x = derive(x)
        return enc_x, dec_x, gt_x

    def __len__(self):
        return self.dataset_length
    
    


