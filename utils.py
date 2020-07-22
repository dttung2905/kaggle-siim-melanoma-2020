from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, List, Optional
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from operator import itemgetter
import numpy as np
import random
import cv2
import os
import albumentations
import pandas as pd
from tqdm import tqdm
from wtfml.engine import Engine


def get_tta_prediction(
    tta: int, test_loader, model, device: str, use_sigmoid: bool, prediction_length: int
) -> np.array:
    """
    get prediction with tta
    """
    prediction = np.zeros(prediction_length)
    for tta_id in range(tta):
        predictions_tta = Engine.predict(
            test_loader, model, device=device, use_sigmoid=True
        )

        predictions_tta = np.vstack((predictions_tta)).ravel()
        prediction += predictions_tta
    prediction /= tta
    return prediction


##################################
## parts of code from catalyst: ##
##################################
def get_OHE_anatom_site(row):
    mapping = {
        "torso": 0,
        "lower_extremity": 1,
        "upper_extremity": 2,
        "head_neck": 3,
        "palms_soles": 4,
        "oral_genital": 5,
    }

    row["site_torso"] = 1 if row["anatom_site_general_challenge"] == "torso" else 0
    row["site_lower_extremity"] = (
        1 if row["anatom_site_general_challenge"] == "lower_extremity" else 0
    )
    row["site_head_neck"] = (
        1 if row["anatom_site_general_challenge"] == "head_neck" else 0
    )
    row["site_palms_soles"] = (
        1 if row["anatom_site_general_challenge"] == "palms_soles" else 0
    )
    return row


def get_meta_feature(input_df):
    tqdm.pandas()
    # Sex features
    input_df["sex"] = input_df["sex"].map({"male": 1, "female": 0})
    input_df["sex"] = input_df["sex"].fillna(-1)

    # Age features
    input_df["age_approx"] /= input_df["age_approx"].max()
    input_df["age_approx"] = input_df["age_approx"].fillna(0)

    input_df["patient_id"] = input_df["patient_id"].fillna(0)
    input_df["anatom_site_general_challenge"] = input_df[
        "anatom_site_general_challenge"
    ].str.replace("/", "_")
    input_df["anatom_site_general_challenge"] = input_df[
        "anatom_site_general_challenge"
    ].str.replace(" ", "_")

    # input_df = pd.concat([input_df, atom_site_OHE], axis=1)
    input_df = input_df.progress_apply(lambda x: get_OHE_anatom_site(x), axis=1)
    meta_columns = ["sex", "age_approx"] + [
        col for col in input_df.columns if "site_" in col
    ]
    meta_columns.remove("anatom_site_general_challenge")
    print("##############################")
    print(input_df.shape)
    return input_df, meta_columns


class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler: Sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(self, labels: List[int], mode: str = "downsampling"):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the datasety
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum() for label in set(labels)}

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode if isinstance(mode, int) else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length
