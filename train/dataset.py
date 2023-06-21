from torch.utils.data import Dataset
from bartti.transform import Noise
import typing


class Dataset_Bart(Dataset):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin", interval:float=0.03):
        self.trans = Noise()
        self.train_idx = []
        self.dataset_length = 0
        self.IN = interval
        self.idx_path = index_path
        self.data_path = data_path
        self.f_data = open(self.data_path, 'rb')
        for line in open(self.idx_path, 'rb'):
            line = line.decode()
            self.train_idx.append([line[0], int(line[1]), int(line[2])])
            self.dataset_length += 1

    def _update(self, data_list: list, car_dict: dict, item: float) -> typing.Tuple[list, dict]:
        data_list.append([self.IN, self.IN, self.IN, self.IN, self.IN])
        car_dict.setdefault(item, set())
        return data_list, car_dict

    def _form_frames(self, info: str) -> typing.Tuple[list, dict]:
        data_list = []
        car_dict = {}
        lines = info.split()
        for line in lines:
            line = line.split(',')
            line_data = []
            for i in range(len(line)):
                item = float(line[i])
                if (i == 0) and (item not in car_dict):
                    data_list, car_dict = self._update(data_list, car_dict, item)
                if i == 1:
                    car_dict[line_data[0]].add(item)
                if i != 0:
                    line_data.append(float(item))
            data_list.append(line_data)
        return data_list, car_dict

    def _intersection(self, car_dict: dict) -> set:
        for k in car_dict:
            if intersection is None:
                intersection = car_dict[k]
            else:
                intersection = intersection.intersection(car_dict[k])
        return intersection.add(self.IN)

    def _select_continue_car(self, car_set: set, data_list: list) -> list:
        for line in data_list:
            if line[0] not in car_set:
                data_list.remove(line)
        return data_list

    def _trans_to_array(self, info: str) -> list:
        data_list, car_dict = self._form_frames(info)
        continue_car = self._intersection(car_dict)
        return self._select_continue_car(continue_car, data_list)

    def __getitem__(self, index: int):
        head, tail = self.train_idx[index][1], self.train_idx[index][2]
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        x = self._trans_to_array(info)
        enc_x, enc_mark, dec_x, dec_mark, gt_x = self.trans.derve(x)  # 这个是用来随机挖空的
        return enc_x, enc_mark, dec_x, dec_mark, gt_x

    def __len__(self):
        return self.dataset_length
