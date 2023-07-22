
from torch.utils.data import Dataset
import typing


class Dataset_Base(Dataset):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin", max_seq_len: int=512, rate:float=0.1):
        self.max_car_num = int(max_seq_len / 5) - 1
        self.train_idx = []
        self.dataset_length = 0
        self.rate = rate
        self.idx_path = index_path
        self.data_path = data_path
        self.f_data = open(self.data_path, 'rb')
        for line in open(self.idx_path, 'rb'):
            line = line.decode().split()[0].split(',')
            self.train_idx.append([line[0], int(line[1]), int(line[2])])
            self.dataset_length += 1

    def _form_frames(self, info: str) -> typing.Tuple[list, dict]:
        data_list = []
        car_dict = {}
        lines = info.split()
        for line in lines:
            line = line.decode().split(',')
            line_data = []
            for i in range(len(line)):
                item = float(line[i])
                if i == 0:
                    car_dict.setdefault(item, set())  # {frame: set()}
                if i == 1:
                    car_dict[float(line[0])].add(item)
                line_data.append(item)
            data_list.append(line_data)
        return data_list, car_dict

    def _intersection(self, car_dict: dict, intersection=None) -> set:
        for k in car_dict:
            if intersection is None:
                intersection = car_dict[k]
            else:
                intersection = intersection.intersection(car_dict[k])
        new_inter = set()
        for k in intersection:
            new_inter.add(k)
            if len(new_inter) >= self.max_car_num - 1:
                break
        return new_inter

    def _select_continue_car(self, car_set: set, data_list: list) -> list:
        new_data = []
        for line in data_list:
            if line[1] in car_set:
                new_data.append(line)
        return new_data

    def _trans_to_array(self, info: str) -> list:
        data_list, car_dict = self._form_frames(info)
        continue_car = self._intersection(car_dict)
        return self._select_continue_car(continue_car, data_list), continue_car

    def _process(self):
        raise NotImplementedError
        return None

    def __getitem__(self, index: int):
        """
        return:
        enc_x, enc_mark, enc_car, dec_x, dec_mark, enc_car, gt_x
        """
        head, tail = self.train_idx[index][1], self.train_idx[index][2]
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        x, car_set = self._trans_to_array(info)  # frame, id, x, y, w, h
        return self._process(x, car_set)  # 这个是用来转换finetune任务格式的

    def __len__(self):
        return self.dataset_length
