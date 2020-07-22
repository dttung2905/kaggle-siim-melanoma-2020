import torch.nn as nn
import torch
from catalyst.contrib.nn.criterion.focal import FocalLossBinary
from loss.calculate_loss import get_loss_value
import pretrainedmodels
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import models


def get_model(
    backbone: str,
    model_name: str,
    num_classes: int,
    input_size: int,
    use_metadata: bool = False,
    meta_features: int = None,
):
    """
    Args:
        backbone: name of the model backbone eg: efficientnet
        model_name: name of the model eg: efficientnet-b7
        num_classes: number of output classes
        input_size: image input size
        use_metadata: whether to use metdata or not
        meta_features: number of meta_features to be used

    Returns:
        model class
    """
    model_backbone_zoo = ["efficientnet", "SEResnext50_32x4d", "resnet"]
    assert backbone in model_backbone_zoo

    if use_metadata:
        assert meta_features is not None

    if backbone == "SEResnext50_32x4d":
        model = SEResnext50_32x4d(pretrained=model_name, num_classes=num_classes)
    elif backbone == "efficientnet":
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        # model = efficientnetBackBone(model_name=model_name, num_classes=num_classes)
    elif backbone == "resnet":
        resnet_zoo = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        assert model_name in resnet_zoo
        model = PretrainedModel(pretrained=model_name, num_classes=num_classes)

    if use_metadata:
        model = ModelWithMetaData(arch=model, n_meta_features=len(meta_features))
    else:
        print("NOT USING METADATA")
    return model


class ModelWithMetaData(nn.Module):
    def __init__(self, arch, n_meta_features: int) -> None:
        """
        this is the model class that uses pretrained backbone & tabular metadata
        Args:
            arch: backbone architecture
            n_meta_features: number of metadata features
        """
        super(ModelWithMetaData, self).__init__()
        self.arch = arch
        in_features = self.arch._fc.in_features
        self.arch._fc = nn.Linear(in_features=in_features, out_features=500, bias=True)
        self.meta = nn.Sequential(
            nn.Linear(n_meta_features, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 250),  # FC layer output will have 250 features
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.ouput = nn.Linear(500 + 250, 1)

    def forward(self, image, targets, meta):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        cnn_features = self.arch(image, targets)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        x = self.ouput(features)
        loss = get_loss_value(x, targets)
        return x, loss


class SEResnext50_32x4d(nn.Module):
    def __init__(self, num_classes: int, pretrained="imagenet"):
        super(SEResnext50_32x4d, self).__init__()
        self.num_classes = num_classes
        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
            pretrained=None
        )
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self._fc = nn.Linear(2048, self.num_classes)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        out = self._fc(x)
        if out.shape[1] == self.num_classes:
            # if not using metadata we will calculate and return loss here
            loss = get_loss_value(out, targets)
            return out, loss
        return out


class PretrainedModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: str):
        super(PretrainedModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.base_model = pretrainedmodels.__dict__[self.pretrained](pretrained=None)
        in_features = self.base_model.last_linear.in_features
        self._fc = nn.Linear(in_features, self.num_classes)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        out = self._fc(x)
        if out.shape[1] == self.num_classes:
            # if not using metadata we will calculate and return loss here
            loss = get_loss_value(out, targets)
            return out, loss
        return out


class efficientnetBackBone(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained(model_name)
        in_features = self.base_model._fc.in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.3)
        self._fc = nn.Linear(in_features, self.num_classes)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = self.dropout(x)
        out = self._fc(x)
        if out.shape[1] == self.num_classes:
            # if not using metadata we will calculate and return loss here
            loss = get_loss_value(out, targets)
            return out, loss
        return out
