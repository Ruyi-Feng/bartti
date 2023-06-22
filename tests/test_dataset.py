from train.dataset import Dataset_Bart
from torch.utils.data import DataLoader


def test_dataset():
    dataset = Dataset_Bart()
    test_loader = DataLoader(dataset, batch_size=4)
    for i, (enc_x, enc_mark, dec_x, dec_mark, gt_x) in enumerate(test_loader):
        # enc_x: list(size=[seq_len, c_in, batch]) -> torch(size=[batch, seq_len, c_in])
        # enc_mark: list(size=[5, batch]) -> torch(size=[batch, 5])
        print("enc_x", enc_x.shape)
        print("enc_mark", enc_mark.shape)
        print("enc_x", dec_x.shape)
        print("enc_mark", dec_mark.shape)
        break

if __name__ == '__main__':
    test_dataset()
