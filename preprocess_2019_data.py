import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="source data")
    parser.add_argument("--target", type=str, required=True, help="target data")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train = pd.read_csv(args.source)
    train["image_name"] = "isic_2019/" + train["image"]
    train["target"] = train["MEL"]
    train = train[train["target"] == 1][["image_name", "target"]]
    train.reset_index(inplace=True, drop=True)

    train.to_csv(args.target, index=False)
