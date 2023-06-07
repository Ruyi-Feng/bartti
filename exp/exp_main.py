
from exp.exp_basic import Exp_Basic
from exp.dataset import Dataset_Bart
from models import Bart
from torch.utils.data import DataLoader
from utils.metrics import metric

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def __build_model(self):
        model = Bart.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, param):
        data_set = Dataset_Bart(**param)
        shuffle_flag = True if self.args.is_train else False
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=shuffle_flag, num_workers=self.args.num_workers, drop_last=self.args.drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCELoss()
        return criterion

    def _save_model(self, vali_loss, path):
        if self.best_score is None:
            self.best_score = vali_loss
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        elif vali_loss < self.best_score:
            self.best_score = vali_loss
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (enc_x, dec_x, gt_x) in enumerate(train_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                outputs = self.model(enc_x, dec_x)
                loss = criterion(outputs, gt_x.to(self.device))
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data()
        vali_data, vali_loader = self._get_data()

        time_now = time.time()
        train_steps = len(train_loader)
        path = self.args.save_path + '/' + 'checkpoint.pth'

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (enc_x, dec_x, gt_x) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)

                outputs = self.model(enc_x, dec_x)

                loss = criterion(outputs, gt_x.to(self.device))
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # saving model
            self._save_model(vali_loss, path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path))

        return self.model

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'checkpoint.pth')))
        test_data, test_loader = self._get_data()
        self.model.eval()
        outputs = []
        trues = []
        with torch.no_grad():
            for i, (enc_x, dec_x, gt_x) in enumerate(train_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                output = self.model(enc_x, dec_x)
                output = output.detach().cpu().numpy()
                outputs.append(output)
                trues.append(gt_x)
        outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
