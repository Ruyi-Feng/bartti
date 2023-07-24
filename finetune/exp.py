from datetime import timedelta
from finetune.data_factory import gen_dataset
import numpy as np
import os
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from bartti.net import Bart
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from train.utils import metric


class Exp_Ft:
    def __init__(self, args, local_rank=-1):
        print("---------init finetune exp---------------")
        self.args = args
        self.args.save_path = self.args.save_path + self.args.task + '/'
        self.best_score = None
        self.WARMUP = 4000
        self.device = torch.device('cuda', local_rank)
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(days=1))
        self.model = self._build_model()

    def _static_param(self, model):
        # 冻结参数的方式
        print("###### static param #######")
        for param in model.enc_embeding.parameters():
            param.requires_grad = False
        for param in model.bart.encoder.parameters():
            param.requires_grad = False
        return model

    def _build_model(self):
        model = Bart(self.args).float().to(self.device)
        print(self.args.save_path)
        if os.path.exists(self.args.save_path + 'checkpoint_best.pth'):
            print("load checkpoints best", self.args.save_path)
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_best.pth', map_location=torch.device('cpu')))
        elif os.path.exists(self.args.save_path + 'checkpoint_last.pth'):
            print("load checkpoints last", self.args.save_path)
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_last.pth', map_location=torch.device('cpu')))
        model = self._static_param(model)
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _get_data(self):
        data_set = gen_dataset(self.args)
        batch_sz = self.args.batch_size
        sampler = None
        drop_last = False
        if self.args.is_train:
            sampler = DistributedSampler(data_set)
            drop_last=self.args.drop_last
        data_loader = DataLoader(data_set, batch_size=batch_sz, sampler=sampler, drop_last=drop_last, pin_memory=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        lamda1 = lambda step: 1 / np.sqrt(max(step , self.WARMUP))
        scheduler = optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lamda1, last_epoch=-1)
        return model_optim, scheduler

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
            for i, (enc_x, dec_x, gt_x) in enumerate(vali_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                gt_x = gt_x.float().to(self.device)
                outputs, loss = self.model(enc_x, dec_x, gt_x)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self):
        _, train_loader = self._get_data()
        vali_data, vali_loader = self._get_data()

        train_steps = len(train_loader)
        path = self.args.save_path + 'checkpoint_'

        model_optim, scheduler = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []

            self.model.train()

            for i, (enc_x, dec_x, gt_x) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                gt_x = gt_x.float().to(self.device)

                _, loss = self.model(enc_x, dec_x, gt_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    iter_count = 0

                loss.backward()
                model_optim.step()
                scheduler.step()

            if dist.get_rank() == 0:
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

                # saving model
                self._save_model(vali_loss, path + 'best.pth')
            dist.barrier()
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(), path + 'last.pth')
        dist.destroy_process_group()

    def test(self):
        if dist.get_rank() == 0:
            test_data, test_loader = self._get_data()
            self.model.eval()
            result = dict()
            with torch.no_grad():
                for i, (enc_x, dec_x, gt_x) in enumerate(test_loader):
                    if i % 100 == 0:
                        print("test: %d"%i)
                    enc_x = enc_x.float().to(self.device)
                    dec_x = dec_x.float().to(self.device)
                    output, loss = self.model(enc_x, dec_x, gt_x, infer=True)
                    output = output.detach().cpu().tolist()
                    gt_x = gt_x.detach().cpu().tolist()
                    result.update({i: {"gt": gt_x, "pd": output}})
            # outputs = np.array(outputs)
            # outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
            # trues = np.array(trues)
            # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            # mae, mse, rmse, mape, mspe = metric(outputs, trues)
            # print('mse:{}, mae:{}'.format(mse, mae))
            return result
