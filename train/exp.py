from datetime import timedelta
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from train.dataset import Dataset_Bart
from bartti.net import Bart
from torch.utils.data import DataLoader
from train.utils import metric, adjust_learning_rate
from torch.nn.parallel import DistributedDataParallel as DDP



class Exp_Main:
    def __init__(self, args, local_rank=-1):
        self.args = args
        self.best_score = None
        self.device = torch.device('cuda', local_rank)
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(days=1))
        self.model = self._build_model()

    def _build_model(self):
        model = Bart(self.args).float().to(self.device)
        if os.path.exists(self.args.save_path + 'checkpoint_best.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_best.pth', map_location=torch.device('cpu')))
        elif os.path.exists(self.args.save_path + 'checkpoint_last.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_last.pth', map_location=torch.device('cpu')))
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _get_data(self, split: str='train'):
        batch_sz = (self.args.batch_size // dist.get_world_size()) if split == 'val' else self.args.batch_size
        data_set = Dataset_Bart(index_path=self.args.index_path, data_path=self.args.data_path, interval=self.args.interval, max_seq_len=self.args.max_seq_len)
        sampler = None
        if self.args.is_train:
            sampler = DistributedSampler(data_set)
        data_loader = DataLoader(data_set, batch_size=batch_sz, sampler=sampler, drop_last=self.args.drop_last, pin_memory=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _save_model(self, vali_loss, path):
        if self.best_score is None:
            self.best_score = vali_loss
            torch.save(self.model.module.state_dict(), path)
        elif vali_loss < self.best_score:
            self.best_score = vali_loss
            torch.save(self.model.module.state_dict(), path)

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (enc_x, enc_mark, dec_x, dec_mark, gt_x) in enumerate(vali_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                frm_mark = torch.zeros((1, self.args.max_seq_len, 1)).float().to(self.device)
                outputs, loss = self.model((enc_x, frm_mark), enc_mark, (dec_x, frm_mark), dec_mark, gt_x)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data()
        vali_data, vali_loader = self._get_data('val')

        time_now = time.time()
        train_steps = len(train_loader)
        path = self.args.save_path + 'checkpoint_'

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (enc_x, enc_mark, dec_x, dec_mark, gt_x) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                gt_x = gt_x.float().to(self.device)
                frm_mark = torch.zeros((1, self.args.max_seq_len, 1)).float().to(self.device)

                _, loss = self.model((enc_x, frm_mark), enc_mark, (dec_x, frm_mark), dec_mark, gt_x)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if dist.get_rank() == 0:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

                # saving model
                self._save_model(vali_loss, path + 'best.pth')
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            dist.barrier()
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(), path + 'last.pth')
        dist.destroy_process_group()

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'checkpoint_best.pth')))
        test_data, test_loader = self._get_data()
        self.model.eval()
        outputs = []
        trues = []
        with torch.no_grad():
            for i, (enc_x, enc_mark, dec_x, dec_mark, gt_x) in enumerate(test_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                frm_mark = torch.zeros((1, self.args.max_seq_len, 1)).float().to(self.device)
                outputs, loss = self.model((enc_x, frm_mark), enc_mark, (dec_x, frm_mark), dec_mark, gt_x)
                output = output.detach().cpu().numpy()
                outputs.append(output)
                trues.append(gt_x)
        outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe = metric(outputs, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

