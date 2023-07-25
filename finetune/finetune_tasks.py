from finetune.dataset_base import Dataset_Base
import random


class Data_Compensation(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin", max_seq_len: int=512):
        super(Data_Compensation, self).__init__(index_path, data_path, max_seq_len)

    def _process(self, x, car_set):
        # [[frame, id, x, y, w, h], ...]
        # 选 r% 删除中间3帧
        del_num = int(len(car_set) * self.rate)
        del_id = random.sample(car_set, del_num)
        frm_head = x[0][0]
        new_x = []
        frm_preserve = [frm_head, frm_head + 4]
        for item in x:
            if (item[1] in del_id) and (item[2] not in frm_preserve):
                continue
            new_x.append(item)
        return new_x


class Data_Prediction(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin", max_seq_len: int=512):
        super(Data_Prediction, self).__init__(index_path, data_path, max_seq_len)

    def _process(self, x, car_set):
        # [[frame, id, x, y, w, h], ...]
        # 选 r% 删除后面3帧
        del_num = int(len(car_set) * self.rate)
        del_id = random.sample(car_set, del_num)
        frm_head = x[0][0]
        new_x = []
        frm_preserve = [frm_head, frm_head + 1]
        for item in x:
            if (item[1] in del_id) and (item[0] not in frm_preserve):
                continue
            new_x.append(item)
        return new_x


class Data_Simulation(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin", data_path:str=".\\data\\data.bin", max_seq_len: int=512):
        super(Data_Simulation, self).__init__(index_path, data_path, max_seq_len)

    def _process(self, x, car_set):
        # [[frame, id, x, y, w, h], ...]
        # 选删除后面2帧
        frm_head = x[0][0]
        new_x = []
        frm_preserve = [frm_head, frm_head + 1, frm_head + 2]
        for item in x:
            if item[0] not in frm_preserve:
                continue
            new_x.append(item)
        return new_x

