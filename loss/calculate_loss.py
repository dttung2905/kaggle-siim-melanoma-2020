import torch.nn as nn
import torch
from catalyst.contrib.nn.criterion.focal import FocalLossBinary
from .roc_auc_loss import RocAucLoss


def get_loss_value(predictions, targets):
    loss_bce = nn.BCEWithLogitsLoss()(
        predictions, targets.view(-1, 1).type_as(predictions)
    )
    loss_focal = FocalLossBinary(gamma=10)(
        predictions, targets.view(-1, 1).type_as(predictions)
    )
    loss = 0.7 * loss_focal + 0.3 * loss_bce

    # loss = RocAucLoss()(predictions, targets)
    return loss
