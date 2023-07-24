from finetune.exp import Exp_Ft
from finetune.params import params as finetune_param
import os
import json


class Evaluation:
    def __init__(self, local_rank, if_pretrain: bool=False, task: str="compensation", rslt_path:str="./results/rslt.json", check_list:list=[]):
        self.args = finetune_param()
        self.args.data_path = './data/val/data.bin'
        self.args.index_path = './data/val/index.bin'
        self.args.batch_size = 1
        self.args.is_train = False
        self.args.task = task
        if if_pretrain:
            self.args.save_path = './checkpoints/pretrain/'
        else:
            self.args.save_path = './checkpoints/finetune/'
        self._obtain_results(rslt_path, local_rank)
        self._check_results(check_list, rslt_path)

    def _obtain_results(self, rslt_path, local_rank):
        print("---------obtain finetune results---------------")
        bartti = Exp_Ft(self.args, local_rank)
        rslt = bartti.test()
        self._save(rslt, rslt_path)

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
        with open(rslt_path, "w") as f:
            rslt = json.dump(rslt, f)

    def _check_results(self, check_list, rslt_path):
        if "overlap" in check_list:
            self._check_overlap(rslt_path)
        if "spd_stable" in check_list:
            self._check_spd_stable(rslt_path)
        if "simulate" in check_list:
            self._check_simulate(rslt_path)

    def _filt(self, gt, pd, enc):
        gt_new, pd_new, enc_new = [], [], []
        for i in range(len(gt)):
            if round(gt[i][0]) == 0:
                break
            gi = [round(gt[i][0]), round(gt[i][1])] + gt[i][2:]
            pi = [round(pd[i][0]), round(pd[i][1])] + pd[i][2:]
            enc = [round(enc[i][0]), round(enc[i][1])] + enc[i][2:]
            gt_new.append(gi)
            pd_new.append(pi)
            enc_new.append(enc)
        return gt_new, pd_new

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
        last_frm = 0
        overlap = 0
        for line in pd:
            if line[0] != last_frm:
                overlap += self._overlaps(boxes)
                boxes = []
                last_frm = line[0]
            boxes.append(line)
        overlap += self._overlaps(boxes)
        return overlap


    def _check_overlap(self, save_path):
        info = self.load(save_path)
        overlap = 0
        for k, values in info.items():
            gt, pd, enc = self._filt(values["gt"], values["pd"], values["enc"])
            overlap += self._count_overlap(pd)
        return overlap

    def _check_spd_stable(self, save_path):
        pass

    def _check_simulate(self, save_path):
        pass

