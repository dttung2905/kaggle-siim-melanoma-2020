import pandas as pd
import numpy as np
from sklearn import model_selection
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="source data")
    parser.add_argument("--target", type=str, required=True, help="target data")
    return parser.parse_args()


def get_fold(row):
    return row["tfrecord"] % 5


if __name__ == "__main__":
    """
    data taken from here
    https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526
    """
    args = get_args()
    # create folds
    df = pd.read_csv(args.source)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    df = df[df["tfrecord"] != (-1)]
    df["kfold"] = df.apply(lambda x: get_fold(x), axis=1)
    df.to_csv(args.target, index=False)
