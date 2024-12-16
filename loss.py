import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, loss_weight=0, batch=True):
        super(dice_bce_loss, self).__init__()
        self.weight =loss_weight
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1.0  # 避免除数为0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def forward(self, pred, mask, seg_pred, seg_mask):
        dice_loss = self.soft_dice_loss(seg_pred, seg_mask)
        bce_loss = self.bce_loss(seg_pred, seg_mask)
        sum_loss = dice_loss + bce_loss
        zero_tensor = torch.tensor(0.0, device=dice_loss.device)  # 创建与 dice_loss 同样设备上的零张量
        return sum_loss, zero_tensor, dice_loss, bce_loss


class l1_loss(nn.Module):
    def __init__(self, loss_weight=0, batch=True):
        super(l1_loss, self).__init__()
        self.weight = loss_weight
        self.batch = batch

    def forward(self,pred, mask, seg_pred, seg_mask):
        # L1损失
        l1_loss = torch.mean(torch.abs(pred - mask))
        return l1_loss, l1_loss, l1_loss, l1_loss


class mse_loss(nn.Module):
    def __init__(self, delta=20, batch=True):
        super(mse_loss, self).__init__()
        self.batch = batch
        self.mse_loss = nn.MSELoss()
        self.delta = delta

    def clamp(self, delta, x):
        return torch.clamp(x, min=-delta, max=delta) / delta

    def forward(self,y_true_sdf, y_pred_sdf, y_true, y_pred):
        # loss = self.mse_loss(self.clamp(self.delta, y_pred), self.clamp(self.delta, y_true))
        y_pred_sdf = torch.tanh(y_pred_sdf)
        loss = self.mse_loss(y_pred_sdf, y_true_sdf)
        return loss*10000, loss, loss, loss


class l2_Dice_Loss(nn.Module):
    def __init__(self, weight=1, batch=True):
        super(l2_Dice_Loss, self).__init__()
        self.weight_l2 = weight
        self.weight_dice = 2 - weight
        self.batch = batch
        self.mse_loss = nn.MSELoss()  # 使用 L2 损失 (MSE)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # 避免除数为0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        dice_score = (2. * intersection + smooth) / (i + j + smooth)
        return dice_score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        dice_loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return dice_loss

    def forward(self, y_true, y_pred):
        # L2损失
        l2_loss = self.mse_loss(y_pred, y_true)
        # Dice损失
        dice_loss = self.soft_dice_loss(y_true, y_pred)
        # 综合损失
        total_loss = self.weight_l2 * l2_loss + self.weight_dice * dice_loss
        return total_loss


class sdf_dice_loss(nn.Module):
    def __init__(self, delta=40, batch=True):
        super(sdf_dice_loss, self).__init__()
        self.batch = batch
        self.mse_loss = nn.MSELoss()
        self.delta = delta

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1.0  # 避免除数为0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        dice_score = (2. * intersection + smooth) / (i + j + smooth)
        return dice_score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        dice_loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return dice_loss

    def clamp(self, delta, x):
        return torch.clamp(x, min=-delta, max=delta)

    def forward(self, y_true_sdf, y_pred_sdf, y_true_seg, y_pred_seg):
        # SDF Loss (MSE 仅用于边界对齐)
        # sdf_loss = self.mse_loss(self.clamp(self.delta, y_pred_sdf), self.clamp(self.delta, y_true_sdf))/self.delta
        sdf_loss = self.mse_loss(y_pred_sdf, y_true_sdf)

        # 分割损失
        dice_loss = self.soft_dice_loss(y_pred_seg, y_true_seg)

        # 组合损失
        total_loss = sdf_loss + dice_loss  # 可以调整两个损失的权重
        return total_loss


class mse_bce_loss(nn.Module):
    def __init__(self, loss_weight=0, batch=True):
        super(mse_bce_loss, self).__init__()
        self.batch = batch
        self.weight = loss_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, mask, seg_pred, seg_mask):
        pred = torch.tanh(pred)
        mse_loss = self.mse_loss(pred, mask)
        mse_loss = mse_loss*self.weight

        # seg loss
        bce_loss = self.bce_loss(seg_pred, seg_mask)

        # sum loss
        total_loss = mse_loss + bce_loss
        zero_loss = torch.tensor(0.0, device=mse_loss.device)  # create a zero tensor on the same device as mse_loss
        return total_loss, mse_loss, zero_loss, bce_loss


class l1_dice_bce_loss(nn.Module):
    def __init__(self, loss_weight=0, batch=True):
        super(l1_dice_bce_loss, self).__init__()
        self.batch = batch
        self.weight = loss_weight
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1.0  # 避免除数为0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def clamp(self, delta, x):
        return torch.clamp(x, min=-delta, max=delta)

    def forward(self, pred, mask, seg_pred, seg_mask):
        # sdf_loss = self.mse_loss(self.clamp(self.delta, y_pred_sdf), self.clamp(self.delta, y_true_sdf))/self.delta
        pred = torch.tanh(pred)
        l1_loss = torch.mean(torch.abs(pred - mask))
        l1_loss = l1_loss*self.weight

        # 分割损失
        dice_loss = self.soft_dice_loss(seg_pred, seg_mask)
        bce_loss = self.bce_loss(seg_pred, seg_mask)

        # 组合损失
        total_loss = l1_loss + dice_loss + bce_loss   # 可以调整三个损失的权重
        return total_loss, l1_loss, dice_loss, bce_loss


class mse_dice_bce_loss(nn.Module):
    def __init__(self, loss_weight=0, batch=True):
        super(mse_dice_bce_loss, self).__init__()
        self.batch = batch
        self.weight = loss_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1.0  # 避免除数为0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    # def clamp(self, delta, x):
    #     return torch.clamp(x, min=-delta, max=delta)

    def forward(self, pred, mask, seg_pred, seg_mask):
        # sdf_loss = self.mse_loss(self.clamp(self.delta, y_pred_sdf), self.clamp(self.delta, y_true_sdf))/self.delta
        mse_loss = self.mse_loss(pred, mask)
        mse_loss = mse_loss*self.weight

        # 分割损失
        dice_loss = self.soft_dice_loss(seg_pred, seg_mask)
        bce_loss = self.bce_loss(seg_pred, seg_mask)

        # 组合损失
        total_loss = mse_loss + dice_loss + bce_loss
        return total_loss, mse_loss, dice_loss, bce_loss
