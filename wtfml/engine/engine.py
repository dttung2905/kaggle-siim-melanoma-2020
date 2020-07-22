import torch
from tqdm import tqdm
from ..utils import AverageMeter
import wandb
import sys

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

try:
    from torch.cuda.amp import GradScaler, autocast

    torch15_available = True
except ImportError:
    torch15_available = False


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
        wandb=False,
        scaler=False,
    ):
        if use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and not torch15_available:
            raise Exception(
                "You want to use fp16 but you dont have pytorch > 1.5 installed"
            )
        if fp16 and use_tpu:
            raise Exception("amp pytorch fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()

            if fp16 and scaler:
                with autocast():
                    _, loss = model(**data)
            elif fp16:
                raise Exception("you mush pass amp scaler in Engine.train method")
            else:
                _, loss = model(**data)

            if not use_tpu:
                with torch.set_grad_enabled(True):
                    if fp16 and scaler:
                        scaler.scale(loss).backward()
                    elif fp16:
                        raise Exception(
                            "you mush pass amp scaler in Engine.train method"
                        )
                    else:
                        loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        if fp16 and scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        elif fp16:
                            raise Exception(
                                "you mush pass amp scaler in Engine.train method"
                            )
                        else:
                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)

        if wandb:
            wandb.log({"train_loss": losses.avg})
        return losses.avg

    @staticmethod
    def evaluate(
        data_loader,
        model,
        device,
        epoch,
        use_tpu=False,
        wandb=False,
        upload_image=False,
        use_sigmoid=False,
    ):
        losses = AverageMeter()
        final_predictions = []
        examples_images = []
        model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)
                if use_sigmoid:
                    predictions = torch.sigmoid(predictions)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
                if wandb and upload_image:
                    examples_batch = [
                        wandb.Image(
                            image,
                            caption=f"epoch: {epoch}, pred: {prediction.item()}, target: {target}",
                        )
                        for image, prediction, target in zip(
                            data["image"], predictions, data["targets"]
                        )
                    ]
                    wandb.log({"Examples": examples_batch})

        if wandb:
            wandb.log({"valid_loss": losses.avg})
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False, use_sigmoid=True):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                if use_sigmoid:
                    predictions = torch.sigmoid(predictions)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions
