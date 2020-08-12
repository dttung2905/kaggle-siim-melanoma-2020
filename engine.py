import pandas as pd

import albumentations
import os
from wtfml.data_loaders.image.classification import ClassificationDataset
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping
import torch
import numpy as np
from sklearn import metrics
import wandb
import random
from class_activation_map import show_class_activation_map
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import yaml
import argparse
from utils import BalanceClassSampler, get_meta_feature, get_tta_prediction
from augmentation.hair import AdvancedHairAugmentation
from augmentation.microscope import Microscope
from augmentation.augmix import RandomAugMix
import sys
from torchcontrib.optim import SWA
from torch.cuda.amp import GradScaler
from models.model_zoo import get_model
from wtfml.logger import logger
from optimizer.optimizer_zoo import RAdam 

def seed_everything(seed=5000):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config_file")
    return parser.parse_args()


def train(fold):
    args = get_args()
    with open(args.config) as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(
        project="siim2020",
        entity="siim_melanoma",
        # name=f"20200718-effb0-adamw-consineaneal-{fold}",
        name=f"2017-2018-rexnet-test-{fold}",
        #name=f"swav-test-{fold}",
        #name=f"RAdam-b6-384x384-{fold}"
    )
    config = wandb.config  # Initialize config
    config.update(config_file)
    device = config.device

    model_path = config.model_path.format(fold)

    seed_everything(config.seed)
    df = pd.read_csv(config.train_csv_fold)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_train["image_name"] = config.training_data_path + df_train["image_name"] + ".jpg"

    if config.supplement_data["use_supplement"]:
        print(f"training shape before merge {df_train.shape}")
        df_supplement = pd.read_csv(config.supplement_data["csv_file"])
        df_supplement = df_supplement[df_supplement["tfrecord"] % 2 == 0]
        df_supplement = df_supplement[df_supplement["target"] == 1]
        df_supplement["image_name"] = (
            config.supplement_data["file_path"] + df_supplement["image_name"] + ".jpg"
        )
        df_train = pd.concat([df_train, df_supplement]).reset_index(drop=True)
        df_train = df_train.sample(frac=1, random_state=config.seed).reset_index(
            drop=True
        )
        del df_supplement
        print(f"training shape after merge {df_train.shape}")

    df_valid = df[df.kfold == fold].reset_index(drop=True)
    df_valid["image_name"] = config.training_data_path + df_valid["image_name"] + ".jpg"

    if config.use_metadata:
        df_train, meta_features = get_meta_feature(df_train)
        df_valid, _ = get_meta_feature(df_valid)
    else:
        meta_features = None

    model = get_model(
        config.model_backbone,
        config.model_name,
        config.num_classes,
        config.input_size,
        config.use_metadata,
        meta_features,
    )

    model = model.to(config.device)
    print("watching model")
    wandb.watch(model, log="all")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            AdvancedHairAugmentation(hairs_folder="../input/melanoma-hairs/"),
            # albumentations.augmentations.transforms.CenterCrop(64, 64, p=0.8),
            albumentations.augmentations.transforms.RandomBrightnessContrast(),
            albumentations.augmentations.transforms.HueSaturationValue(),
            # Microscope(p=0.4),
            albumentations.augmentations.transforms.RandomResizedCrop(
                config.input_size, config.input_size, scale=(0.7, 1.0), p=0.4
            ),
            albumentations.augmentations.transforms.VerticalFlip(p=0.4),
            albumentations.augmentations.transforms.Cutout(p=0.3), # doesnt work
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15
            ),
            albumentations.Flip(p=0.5),
            RandomAugMix(severity=7, width=7, alpha=5, p=0.3),
            # albumentations.augmentations.transforms.Resize(
            #    config.input_size, config.input_size, p=1
            # ),
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
        ]
    )

    valid_aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),]
    )

    train_images = df_train.image_name.values.tolist()
    # train_images = [
    #    os.path.join(config.training_data_path, i + ".jpg") for i in train_images
    # ]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    # valid_images = [
    #    os.path.join(config.training_data_path, i + ".jpg") for i in valid_images
    # ]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
        meta_features=meta_features,
        df_meta_features=df_train,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        # num_workers=4,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        #sampler=BalanceClassSampler(labels=train_targets, mode="upsampling"),
        drop_last=True,
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
        meta_features=meta_features,
        df_meta_features=df_valid,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        # num_workers=4,
        num_workers=1,
        pin_memory=True,
        # drop_last=True
    )

    #optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    optimizer = RAdam(model.parameters(), lr=config.lr)
    if config.swa["use_swa"]:
        optimizer = SWA(optimizer, swa_start=12, swa_freq=1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, patience=2, threshold=0.0001, mode="max"
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, len(train_loader) * config.epochs
    # )

    #scheduler = torch.optim.lr_scheduler.CyclicLR(
    #   optimizer,
    #   base_lr=config.lr / 10,
    #   max_lr=config.lr * 100,
    #   mode="triangular2",
    #   cycle_momentum=False,
    #)

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #    optimizer, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=config.epochs
    #)

    es = EarlyStopping(patience=6, mode="max")
    if config.fp16:
        print("************* using fp16 *************")
        scaler = GradScaler()
    else:
        scaler = False

    for epoch in range(config.epochs):
        train_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device=config.device,
            wandb=wandb,
            accumulation_steps=config.accumulation_steps,
            fp16=config.fp16,
            scaler=scaler,
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device=config.device,
            wandb=wandb,
            epoch=epoch,
            upload_image=False,
            use_sigmoid=True,
        )
        predictions = np.vstack((predictions)).ravel()

        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        wandb.log(
            {"valid_auc": auc,}
        )

        scheduler.step(auc)

        es(auc, model, model_path=model_path)
        if es.early_stop:
            print("Early stopping")
            break
    if config.swa["use_swa"]:
        print("saving the model using SWA")
        optimizer.swap_swa_sgd()
        torch.save(model.state_dict(), config.swa["model_path"].format(fold))

    evaluate_for_best_epoch(
        fold,
        model_path,
        config.device,
        valid_loader,
        config.model_name,
        valid_targets,
        "final",
        meta_features=meta_features,
    )
    if config.swa["use_swa"]:
        model_path = config.swa["model_path"].format(fold)
        evaluate_for_best_epoch(
            fold,
            model_path,
            config.device,
            valid_loader,
            config.model_name,
            valid_targets,
            "swa",
            meta_features=meta_features,
        )

    # show_class_activation_map(model, valid_loader, wandb, 10)


def evaluate_for_best_epoch(
    fold,
    model_path,
    device,
    valid_loader,
    model_name,
    valid_targets,
    epoch="final",
    meta_features=None,
):
    args = get_args()
    with open(args.config) as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)
    config = wandb.config  # Initialize config
    config.update(config_file)

    print(f"Evaluating on epoch {epoch} from {model_path}")
    model = get_model(
        config.model_backbone,
        config.model_name,
        config.num_classes,
        config.input_size,
        config.use_metadata,
        meta_features,
    )
    model.load_state_dict(torch.load(model_path))

    model.to(device, non_blocking=True)
    predictions, valid_loss = Engine.evaluate(
        valid_loader,
        model,
        device=device,
        wandb=wandb,
        epoch=epoch,
        upload_image=True,
        use_sigmoid=True,
    )
    predictions = np.vstack((predictions)).ravel()

    auc = metrics.roc_auc_score(valid_targets, predictions)
    oof_file = config.oof_file.replace(".npy", "_" + str(fold) + ".npy")
    np.save(oof_file, valid_targets)
    print(f"Epoch = {epoch}, AUC = {auc}")
    wandb.log(
        {"best_valid_auc": auc,}
    )


def predict(fold):
    print(f"Prediction on test set fold {fold}")
    args = get_args()
    with open(args.config) as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)
    config = wandb.config  # Initialize config
    config.update(config_file)

    df_test = pd.read_csv(config.test_csv)
    if config.use_metadata:
        df_test, meta_features = get_meta_feature(df_test)
    else:
        meta_features = None

    if config.swa["use_swa"]:
        model_path = config.swa["model_path"].format(fold)
        print(f"using SWA, loading checkpoint from {model_path}")
    else:
        model_path = config.model_path.format(fold)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # aug = albumentations.Compose(
    #    [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    # )
    aug = albumentations.Compose(
        [
            AdvancedHairAugmentation(hairs_folder="../input/melanoma-hairs/"),
            # albumentations.augmentations.transforms.CenterCrop(64, 64, p=0.8),
            albumentations.augmentations.transforms.RandomBrightnessContrast(),
            albumentations.augmentations.transforms.HueSaturationValue(),
            # Microscope(p=0.4),
            albumentations.augmentations.transforms.RandomResizedCrop(
                config.input_size, config.input_size, scale=(0.7, 1.0), p=0.4
            ),
            albumentations.augmentations.transforms.VerticalFlip(p=0.4),
            # albumentations.augmentations.transforms.Cutout(p=0.8), # doesnt work
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15
            ),
            albumentations.Flip(p=0.5),
            # RandomAugMix(severity=7, width=7, alpha=5, p=1),
            # albumentations.augmentations.transforms.Resize(
            #    config.input_size, config.input_size, p=1
            # ),
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
        ]
    )

    images = df_test.image_name.values.tolist()
    images = [os.path.join(config.test_data_path, i + ".jpg") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationDataset(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
        meta_features=meta_features,
        df_meta_features=df_test,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=4,
    )

    model = get_model(
        config.model_backbone,
        config.model_name,
        config.num_classes,
        config.input_size,
        config.use_metadata,
        meta_features,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)

    if config.tta:
        predictions = get_tta_prediction(
            config.tta, test_loader, model, config.device, True, len(images)
        )
    else:
        predictions = Engine.predict(
            test_loader, model, device=config.device, use_sigmoid=True
        )

        predictions = np.vstack((predictions)).ravel()

    return predictions
