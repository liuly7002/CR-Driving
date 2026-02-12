import torch
import torch.nn.functional as F

"""
为什么结合这些损失函数？
类别不平衡：通过 Focal Loss 和加权交叉熵损失，模型可以减少对常见类别（如类别 0 和 4）的偏向，增加对较少类别（如类别 1、2、3）的关注。
等级关系：通过 Ordinal Regression Loss，模型可以更好地理解类别之间的等级关系，避免将高风险类别（如类别 0）误分类为低风险类别（如类别 4）。
强化模型的鲁棒性：通过组合多个损失函数，模型可以同时从多个角度进行学习，提升整体性能。
"""
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, weight=None):
        super(CombinedLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.ordinal_loss = OrdinalRegressionLoss(weight=weight)

        self.ce_weight = 1.0     # Cross Entropy loss的权重
        self.focal_weight = 1.0  # Focal loss的权重
        self.ord_weight = 1.0    # Ordinal loss的权重

    def forward(self, inputs, targets):
        # 计算 Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(inputs, targets)

        # 计算 Focal Loss
        focal_loss = self.focal_loss(inputs, targets)

        # 计算 Ordinal Loss
        ord_loss = self.ordinal_loss(inputs, targets)

        # 组合损失函数
        total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss + self.ord_weight * ord_loss
        return total_loss


# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = (1 - pt) ** self.gamma * CE_loss

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# Ordinal Regression Loss
class OrdinalRegressionLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(OrdinalRegressionLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)

        # 计算类别之间的平滑性
        ord_penalty = torch.abs(inputs[:, 1:] - inputs[:, :-1]).mean()

        return ce_loss + 0.5 * ord_penalty  # 0.5 可调节
