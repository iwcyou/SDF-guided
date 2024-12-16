import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils.metrics import IoU
from loss import dice_bce_loss, l2_Dice_Loss, l1_loss, mse_loss, sdf_dice_loss, mse_bce_loss, l1_dice_bce_loss, mse_dice_bce_loss
import wandb

loss_classes={
    "dice_bce_loss": dice_bce_loss,
    "l2_Dice_Loss": l2_Dice_Loss,
    "l1_loss": l1_loss,
    "mse_loss": mse_loss,
    "sdf_dice_loss": sdf_dice_loss,
    "mse_bce_loss": mse_bce_loss,
    "l1_dice_bce_loss": l1_dice_bce_loss,
    "mse_dice_bce_loss": mse_dice_bce_loss
}

class Solver:
    def __init__(self, net, optimizer, loss_name, metrics=IoU, loss_weight=1.0):
        self.net = net.cuda() #调用model.cuda()，可以将模型加载到GPU上去。但建议使用model.to(device)的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下。
        # self.net = torch.nn.DataParallel(
        #     self.net, device_ids=list(range(torch.cuda.device_count()))) #多个显卡共同计算
        self.net = torch.nn.DataParallel(self.net)
        self.optimizer = optimizer
        self.loss = loss_classes[loss_name](loss_weight = loss_weight)
        self.metrics = metrics(threshold=0.5)
        self.old_lr = optimizer.param_groups[0]["lr"] #？？？

    def set_input(self, img_batch, mask_batch=None, img_id=None): #输入图片
        self.img = img_batch
        self.mask = mask_batch
        self.seg_mask = torch.where(mask_batch > 0, torch.tensor(1.0, device=mask_batch.device), torch.tensor(0.0, device=mask_batch.device)) #将mask_batch中大于0的值变为1，否则为0
        self.img_id = img_id

    def forward(self):
        self.img = self.img.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()
            self.seg_mask = self.seg_mask.cuda()

    def optimize(self): #优化器
        self.net.train()
        self.forward()
        self.optimizer.zero_grad()
        # print(self.img.shape)
        pred = self.net.forward(self.img)
        # seg_pred = (pred>0).detach().int() + (pred-pred.detach())
        seg_pred = torch.sigmoid(pred)
        pred = torch.tanh(pred)
        loss, sdf_loss, dice_loss, bce_loss = self.loss(pred, self.mask, seg_pred, self.seg_mask)
        loss.backward()
        # sdf_loss.backward(retain_graph=True)
        # sdf_grad = {name: param.grad.clone() for name, param in self.net.named_parameters()}
        # dice_loss.backward(retain_graph=True)
        # dice_grad = {name: param.grad.clone() for name, param in self.net.named_parameters()}
        # bce_loss.backward()
        # bce_grad = {name: param.grad.clone() for name, param in self.net.named_parameters()}
        self.optimizer.step()
        metrics = self.metrics(self.seg_mask, seg_pred)
        return seg_pred, loss.item(), metrics, sdf_loss.item(), dice_loss.item(), bce_loss.item()

    def save_weights(self, path): #保留模型参数
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path): #加载模型参数
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False): #更新学习率
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print("==> update learning rate: %f -> %f" % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def test_batch(self):
        self.net.eval() #评估模式
        with torch.no_grad():
            self.forward()
            pred = self.net.forward(self.img)
            seg_pred = torch.sigmoid(pred)
            pred = torch.tanh(pred)
            loss, sdf_loss, dice_loss, bce_loss = self.loss(pred, self.mask, seg_pred, self.seg_mask)
            metrics = self.metrics(self.seg_mask, seg_pred)
            # pred = pred.cpu().data.numpy()
        return seg_pred, loss.item(), metrics, sdf_loss.item(), dice_loss.item(), bce_loss.item()

    def pred_one_image(self, image):
        self.net.eval()
        image = image.cuda().unsqueeze(0)
        pred = self.net.forward(image)
        return pred.cpu().data.numpy().squeeze(1).squeeze(0)

    # def pred_one_image(self, image, mask):
    #     """return pred and IoU"""
    #     self.net.eval()
    #     image = image.cuda().unsqueeze(0)
    #     pred = self.net.forward(image)
    #     metrics = self.metrics(mask, pred)
    #     return pred.cpu().data.numpy().squeeze(1).squeeze(0), metrics[3].item()


class Trainer:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_train_dl(self, dataloader): #装载训练集
        self.train_dl = dataloader

    def set_validation_dl(self, dataloader): #装载验证集
        self.validation_dl = dataloader

    def set_test_dl(self, dataloader): #装载测试集
        self.test_dl = dataloader

    def set_save_path(self, save_path): #加载参数保存路径
        self.save_path = save_path

    def fit_one_epoch(self, dataloader, eval=False):
        dataloader_iter = iter(dataloader) #将dataloader变成iterator
        epoch_loss = 0
        epoch_sdf_loss = 0
        epoch_dice_loss = 0
        epoch_bce_loss = 0
        epoch_metrics = 0
        iter_num = len(dataloader_iter)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num) #进度条
        for i, (img, mask) in progress_bar:
            self.solver.set_input(img, mask)
            if eval:
                pred, iter_loss, iter_metrics, iter_sdf_loss, iter_dice_loss, iter_bce_loss = self.solver.test_batch()
                # wandb.log({"eval_sat":[wandb.Image(img[2][:3])], "eval_pred": [wandb.Image(pred[2])], "eval_mask": [wandb.Image(mask[2])]})
            else:
                pred, iter_loss, iter_metrics, iter_sdf_loss, iter_dice_loss, iter_bce_loss = self.solver.optimize()
                if i == 1:
                    wandb.log({"train_sat":[wandb.Image(img[2][:3])], "train_pred": [wandb.Image(pred[2])], "train_mask": [wandb.Image(mask[2])]})
            epoch_loss += iter_loss
            epoch_sdf_loss += iter_sdf_loss
            epoch_dice_loss += iter_dice_loss
            epoch_bce_loss += iter_bce_loss
            epoch_metrics += iter_metrics
            progress_bar.set_description(
                f'iter: {i} loss: {iter_loss:.4f} metrics: {iter_metrics[3]:.4f}'
            )
        epoch_loss /= iter_num
        epoch_metrics /= iter_num
        epoch_sdf_loss /= iter_num
        epoch_dice_loss /= iter_num
        epoch_bce_loss /= iter_num
        return epoch_loss, epoch_metrics, epoch_sdf_loss, epoch_dice_loss, epoch_bce_loss


    def fit(self, epochs, no_optim_epochs=10):

        val_best_metrics = 0
        test_best_metrics = 0
        val_best_loss = float("+inf") #无穷大
        test_best_loss = float("+inf") #无穷大
        no_optim = 0
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")

            print(f"training")
            train_loss, train_metrics, train_sdf_loss, train_dice_loss, train_bce_loss = self.fit_one_epoch(self.train_dl, eval=False)

            print(f"validating")
            val_loss, val_metrics, val_sdf_loss, val_dice_loss, val_bce_loss = self.fit_one_epoch(self.validation_dl, eval=True)

            print(f"testing")
            test_loss, test_metrics, test_sdf_loss, test_dice_loss, test_bce_loss = self.fit_one_epoch(self.test_dl, eval=True)

            print('epoch finished')
            print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
            print(f'val_loss: {val_loss:.4f} val_metrics: {val_metrics}')
            print(f'test_loss: {test_loss:.4f} test_metrics: {test_metrics}')
            print()

            wandb.log({"epoch": epoch + 1,
                       "train_loss": train_loss, "train_IoU": train_metrics[3],
                       "val_loss": val_loss, "val_IoU": val_metrics[3],
                       "test_loss": test_loss,
                       "train_sdf_loss": train_sdf_loss, "train_dice_loss": train_dice_loss, "train_bce_loss": train_bce_loss,
                       "val_sdf_loss": val_sdf_loss, "val_dice_loss": val_dice_loss, "val_bce_loss": val_bce_loss})

            #记录验证集的最优
            if val_metrics[3] > val_best_metrics:
                val_best_metrics = val_metrics[3]
                self.solver.save_weights(os.path.join(self.save_path,
                    f"epoch{epoch}_val{val_metrics[3]:.4f}_test{test_metrics[0]:.4f}_{test_metrics[1]:.4f}_{test_metrics[2]:.4f}_{test_metrics[3]:.4f}_{test_metrics[4]:.4f}.pth"))
                wandb.log({"test_IoU": test_metrics[3]})

            if val_loss < val_best_loss:
                no_optim = 0
                val_best_loss = val_loss
            else:
                no_optim += 1

            if no_optim > no_optim_epochs:
                if self.solver.old_lr < 1e-8:
                    print('early stop at {epoch} epoch')
                    break
                else:
                    no_optim = 0
                    self.solver.update_lr(5.0, factor=True) #更新学习率
        wandb.finish()


class Tester:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def predict(self):
        pass
