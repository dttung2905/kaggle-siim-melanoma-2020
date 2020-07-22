import torch
import torch.nn.functional as F
import numpy as np


def criterion_margin_focal_binary_cross_entropy(
    logit: torch.Tensor, truth: torch.Tensor,
) -> torch.Tensor:
    """
    Implementation of Soft Marrgin Focal Loss.
    For more information, please refer to:
    https://www.researchgate.net/figure/Comparisons-among-soft-margin-focal-loss-SMFL-the-softmargin-cross-entropy-SMCE_fig2_333372925

    Args:
        logit: Output of Feedforward function
        truth: Ground truth labels

    Returns:
        Computed loss
    
    source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155201#880870
    """
    weight_pos = 2
    weight_neg = 1
    gamma = 2
    margin = 0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid(logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth * log_pos + (1 - truth) * log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em + (1 - em) * prob)

    weight = truth * weight_pos + (1 - truth) * weight_neg
    loss = margin + weight * (1 - prob) ** gamma * log_prob

    loss = loss.mean()
    return loss
