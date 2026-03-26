import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class IoULoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.iou = IoULoss(num_classes=num_classes)

    def forward(self, pred, target):
        loss_dice = self.dice(pred, target)
        loss_iou = self.iou(pred, target)
        loss_total = loss_dice + loss_iou
        return loss_total, loss_dice, loss_iou