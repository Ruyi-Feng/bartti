from finetune.exp import Exp_Ft
from finetune.params import params as finetune_param
import os
import json


class Evaluation:
    def __init__(self, if_pretrain: bool=False, task: str="compensation"):
        self.args = finetune_param()
        self.args.data_path = './data/val/data.bin'
        self.args.index_path = './data/val/index.bin'
        self.args.batch_size = 2000
        self.args.is_train = False
        self.args.task = task
        if if_pretrain:
            self.args.save_path = './checkpoints/pretrain_enc6_dec6/'
        else:
            self.args.save_path = './checkpoints/finetune_enc6_dec6/'
        # self.obtain_results(rslt_path, local_rank)
        # self.check_results(check_list, rslt_path)

    def obtain_results(self, rslt_path, local_rank):
        print("---------obtain finetune results---------------")
        bartti = Exp_Ft(self.args, local_rank)
        rslt = bartti.test()
        self._save(rslt, rslt_path)
        print("finish save results", self.args.task)

    def load(self, flnm) -> dict:
        with open(flnm, 'r') as load_f:
            info = json.load(load_f)
        return info

    def _save(self, rslt, rslt_path):
        """
        rslt: dict
        ----
        {
            i: {"gt": [], "pd": []},
        }
        gt: list
        [split(5 frm)
            [frm, car_id, x, y, w, h],
            [frm, car_id, x, y, w, h],
        ]
        """
        rslt = self._filt(rslt)
        with open(rslt_path, "w") as f:
            rslt = json.dump(rslt, f)

    def check_results(self, check_list: list=[], rslt_path: str="./results/rslt.json"):
        if "mse" in check_list:
            self._check_mse(rslt_path)
        if "overlap" in check_list:
            num, total = self._check_overlap(rslt_path)
            print("------%s-------", self.args.task)
            print("overlap num | total: ", num, total, num/total)
        if "spd_stable" in check_list:
            self._check_spd_stable(rslt_path)
        if "simulate" in check_list:
            self._check_simulate(rslt_path)

    def _stand(self, gt, pd, enc):
        gt_new, pd_new, enc_new = [], [], []
        for i in range(len(gt)):
            if round(gt[i][0]) == 0:
                break
            gi = [round(gt[i][0]), round(gt[i][1])] + gt[i][2:]
            pi = [round(pd[i][0]), round(pd[i][1])] + pd[i][2:]
            enc_i = [round(enc[i][0]), round(enc[i][1])] + enc[i][2:]
            gt_new.append(gi)
            pd_new.append(pi)
            enc_new.append(enc_i)
        return gt_new, pd_new, enc_new

    def _filt(self, rslt):
        # rslt: batch, seq_len, d_model
        new_rslt = dict()
        for k, values in rslt.items():
            for batch in range(len(values["gt"])):
                gt, pd, enc = self._stand(values["gt"][batch], values["pd"][batch], values["enc"][batch])
                new_rslt.update({str(k) + str(batch): {"gt": gt, "pd": pd, "enc": enc}})
        return new_rslt

    def _dis(self, a1, a2, w1, w2):
        return (abs(a1 - a2) - 0.5 * (w1 + w2)) < 0

    def _intersect(self, box1, box2):
        return (self._dis(box1[0], box2[0], box1[2], box2[2]) and self._dis(box1[1], box2[1], box1[3], box2[3]))

    def _overlaps(self, boxes):
        overlap = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                overlap += self._intersect(boxes[i][2:], boxes[j][2:])
        return overlap

    def _count_overlap(self, pd):
        ##### 需要排除预测0值
        last_frm = 0
        overlap = 0
        boxes = []
        for line in pd:
            if line[0] != last_frm:
                if line[0] == 0:
                    break
                overlap += self._overlaps(boxes)
                boxes = []
                last_frm = line[0]
            boxes.append(line)
        overlap += self._overlaps(boxes)
        return overlap

    def _check_overlap(self, save_path):
        info = self.load(save_path)
        """
        info: dict
        {k: {"gt": [], "pd": [], "enc": []}}
        """
        overlap = 0
        total_num = 0
        for k in info:
            total_num += len(info[k]["gt"])
            overlap += self._count_overlap(info[k]["pd"])
        return overlap, total_num

    def _check_mse(self, rslt_path):
        pass

    def _check_spd_stable(self, save_path):
        pass

    def _check_simulate(self, save_path):
        pass

