import torch
import torch.nn.functional as F


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, weight=None, smooth=0.1):
        super(CombinedLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.ordinal_loss = OrdinalRegressionLoss(weight=weight, smooth_factor=0.15)
        self.dice_loss = DiceLoss(weight=weight, smooth=0.2)

        self.register_buffer('running_ce_loss', torch.tensor(0.0))
        self.register_buffer('running_focal_loss', torch.tensor(0.0))
        self.register_buffer('running_ord_loss', torch.tensor(0.0))
        self.register_buffer('running_dice_loss', torch.tensor(0.0))
        self.momentum = 0.95

        self.ce_weight = 0.2
        self.focal_weight = 0.45
        self.ord_weight = 0.15
        self.dice_weight = 0.2

    def forward(self, inputs, targets):

        ce_loss = self.cross_entropy_loss(inputs, targets)
        
        try:
            focal_loss = self.focal_loss(inputs, targets)
        except Exception as e:
            print(f"Warning: Failed to compute focal loss: {e}")
            focal_loss = torch.tensor(0.0, device=inputs.device)
            
        try:
            ord_loss = self.ordinal_loss(inputs, targets)
        except Exception as e:
            print(f"Warning: Failed to compute ordinal loss: {e}")
            ord_loss = torch.tensor(0.0, device=inputs.device)
            
        try:
            dice_loss = self.dice_loss(inputs, targets)
        except Exception as e:
            print(f"Warning: Failed to compute dice loss: {e}")
            dice_loss = torch.tensor(0.0, device=inputs.device)

        self.running_ce_loss = self.momentum * self.running_ce_loss + (1 - self.momentum) * ce_loss.detach()
        self.running_focal_loss = self.momentum * self.running_focal_loss + (1 - self.momentum) * focal_loss.detach()
        self.running_ord_loss = self.momentum * self.running_ord_loss + (1 - self.momentum) * ord_loss.detach()
        self.running_dice_loss = self.momentum * self.running_dice_loss + (1 - self.momentum) * dice_loss.detach()

        total_running_loss = self.running_ce_loss + self.running_focal_loss + self.running_ord_loss + self.running_dice_loss
        alpha = 0.9
        if total_running_loss > 0:
            dynamic_ce_weight = self.ce_weight * (self.running_ce_loss / total_running_loss).clamp(0.1, 0.3)
            dynamic_focal_weight = self.focal_weight * (self.running_focal_loss / total_running_loss).clamp(0.3, 0.5)
            dynamic_ord_weight = self.ord_weight * (self.running_ord_loss / total_running_loss).clamp(0.1, 0.2)
            dynamic_dice_weight = self.dice_weight * (self.running_dice_loss / total_running_loss).clamp(0.15, 0.25)
        else:
            dynamic_ce_weight = self.ce_weight
            dynamic_focal_weight = self.focal_weight
            dynamic_ord_weight = self.ord_weight
            dynamic_dice_weight = self.dice_weight

        total_loss = (
            dynamic_ce_weight * ce_loss + 
            dynamic_focal_weight * focal_loss + 
            dynamic_ord_weight * ord_loss + 
            dynamic_dice_weight * dice_loss
        )
        
        return torch.abs(total_loss)

# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)

        if len(inputs.shape) == 4:
            B, C, H, W = inputs.shape

            inputs = inputs.reshape(B, C, -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.reshape(-1, C)
            targets = targets.reshape(-1)

        targets_one_hot = F.one_hot(targets, num_classes).float()

        smooth_factor = 0.1
        targets_smooth = (1 - smooth_factor) * targets_one_hot + smooth_factor / num_classes

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        class_weights = torch.ones_like(probs)
        class_weights[:, 4] = 1.2

        focal_weight = (1 - probs) ** self.gamma
        focal_weight = focal_weight * class_weights

        loss = -(targets_smooth * log_probs * focal_weight).sum(dim=1)

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_weight = alpha[targets]
            loss = alpha_weight * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# Ordinal Regression Loss
class OrdinalRegressionLoss(torch.nn.Module):
    def __init__(self, weight=None, smooth_factor=0.15):
        super(OrdinalRegressionLoss, self).__init__()
        self.weight = weight
        self.smooth_factor = smooth_factor

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)

        probs = F.softmax(inputs, dim=1)

        if len(inputs.shape) == 4:
            B, C, H, W = inputs.shape
            position_weight = torch.ones(H, W, device=inputs.device)
            position_weight[:H//3, :] = 1.2

            ord_penalty = torch.abs(probs[:, 1:] - probs[:, :-1])

            ord_penalty = (ord_penalty.mean(1) * position_weight).mean()
        else:
            ord_penalty = torch.abs(probs[:, 1:] - probs[:, :-1]).mean()

        if len(inputs.shape) == 4:
            return ce_loss + 0.35 * ord_penalty
        else:
            targets_one_hot = F.one_hot(targets, inputs.size(1)).float()
            target_probs = targets_one_hot * (1 - self.smooth_factor) + self.smooth_factor / inputs.size(1)
            kl_div = F.kl_div(F.log_softmax(inputs, dim=1), target_probs, reduction='batchmean')
            kl_div = torch.abs(kl_div)
            return ce_loss + 0.35 * ord_penalty + 0.25 * kl_div

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, smooth=0.2):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        if len(inputs.shape) == 4:
            targets = targets.long()
            B, C, H, W = inputs.shape

            position_weight = torch.ones(H, W, device=inputs.device)

            position_weight[:H//3, :] = 1.2
            
            dice_scores = []
            for cls in range(num_classes):
                target_mask = (targets == cls).float()
                pred_mask = probs[:, cls]

                weighted_intersection = (pred_mask * target_mask * position_weight).sum()
                weighted_union = ((pred_mask * pred_mask * position_weight).sum() + 
                                (target_mask * target_mask * position_weight).sum())
                
                dice = (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
                dice_scores.append(dice)
            
            dice = torch.stack(dice_scores).mean()
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            intersection = torch.sum(probs * targets_one_hot, dim=0)
            cardinality = torch.sum(probs * probs + targets_one_hot * targets_one_hot, dim=0)
            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice = dice.mean()
        
        dice = torch.clamp(dice, 0.0, 1.0)
        
        if self.weight is not None and len(inputs.shape) != 4:
            weight = self.weight.to(inputs.device)
            dice = (dice * weight).mean() / weight.mean()
            
        return (1.0 - dice).abs()
