from evaluation import Evaluation
import os


if __name__ == '__main__':
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    for tsk in ["compensation", "prediction", "simualtion"]:
        Evaluation(local_rank, if_pretrain=True, task=tsk, check_list=["overlap"], rslt_path="./results/"+tsk+"/rslt.json")


