import torch
import torch.nn as nn


class RocAucLoss(nn.Module):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., &amp; Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """

    # https://github.com/tflearn/tflearn/blob/5a674b7f7d70064c811cbd98c4a41a17893d44ee/tflearn/objectives.py
    def forward(self, y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred)
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        # print(y_true)
        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3
        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        mask = difference > 0
        masked = difference.masked_fill(mask, 0)
        return torch.mean(torch.pow(-masked, p))
