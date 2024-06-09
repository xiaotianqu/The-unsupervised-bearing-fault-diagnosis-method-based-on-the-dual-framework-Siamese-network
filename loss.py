import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        dist = F.pairwise_distance(x1, x2)
        total_loss = (1 - y) * torch.pow(dist, 2) + \
                     y * torch.pow(torch.clamp_min_(self.margin - dist, 0), 2)
        loss = torch.mean(total_loss)
        return loss

def manhattan_distance(x1, x2):
        return torch.sum(torch.abs(x1 - x2), dim=-1)
class ContrastiveLoss_1(torch.nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss_1, self).__init__()
        self.margin = margin


    def forward(self, x1, x2, y):
        dist = manhattan_distance(x1, x2)
        total_loss = (1 - y) * torch.pow(dist, 2) + \
                     y * torch.pow(torch.clamp_min_(self.margin - dist, 0), 2)
        loss = torch.mean(total_loss)
        return loss


class ContrastiveLoss_2(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_2, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        # 计算切比雪夫距离
        dist = torch.abs(x1 - x2).max(dim=1)[0]

        # 计算损失
        total_loss = (1 - y) * torch.pow(dist, 2) + \
                     y * torch.pow(torch.clamp_min(self.margin - dist, 0), 2)

        # 求平均损失
        loss = torch.mean(total_loss)
        return loss
class AdaptiveMargin(torch.nn.Module):
    def __init__(self, margin, margin_scale=0.1):
        super(AdaptiveMargin, self).__init__()
        self.margin = margin
        self.margin_scale = margin_scale

    def forward(self, output1, output2, label):
        return self.adaptive_margin_loss(output1, output2, label, self.margin_scale)

    def adaptive_margin_loss(self, output1, output2, label, margin_scale):
        cos_sim = F.cosine_similarity(output1, output2, dim=1)
        margin = margin_scale * (1 - cos_sim)
        loss = torch.where(label == 1, (output1 - output2).pow(2).sum(1), F.relu(margin - (output1 - output2).pow(2).sum(1)))
        return loss.mean()
