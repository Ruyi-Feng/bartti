from finetune.exp import Exp_Ft
from finetune.params import params as finetune_param
import os


class Evaluation:
    def __init__(self, local_rank, if_pretrain: bool=False, task: str="compensation", rslt_path:str="./results", check_list:list=[]):
        self.args = finetune_param()
        self.args.data_path = './data/val/data.bin'
        self.args.index_path = './data/val/index.bin'
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
        gt, outputs = bartti.test()
        self._save(gt, outputs, rslt_path)

    def _save(self, gt, outputs, rslt_path):
        # 以某种数据格式把gt和output存成便于评估的形式
        pass

    def _check_results(self, check_list, rslt_path):
        if "overlap" in check_list:
            self._check_overlap(rslt_path)
        if "spd_stable" in check_list:
            self._check_spd_stable(rslt_path)
        if "simulate" in check_list:
            self._check_simulate(rslt_path)

    def _check_overlap(self, save_path):
        pass

    def _check_spd_stable(self, save_path):
        pass

    def _check_simulate(self, save_path):
        pass

