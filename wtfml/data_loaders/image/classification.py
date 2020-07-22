import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        meta_features=None,
        df_meta_features=None,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.meta_features = meta_features
        self.df_meta_features = df_meta_features

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.meta_features:
            metadata = np.array(
                self.df_meta_features.iloc[item][self.meta_features].values,
                dtype=np.float32,
            )
            return {
                "image": torch.tensor(image),
                "targets": torch.tensor(targets),
                "meta": torch.tensor(metadata),
            }
        else:
            return {
                "image": torch.tensor(image),
                "targets": torch.tensor(targets),
            }


class ClassificationDataLoader:
    def __init__(self, image_paths, targets, resize, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.dataset = ClassificationDataset(
            image_paths=self.image_paths,
            targets=self.targets,
            resize=self.resize,
            augmentations=self.augmentations,
        )

    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True, tpu=False):
        sampler = None
        if tpu == True:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return data_loader
