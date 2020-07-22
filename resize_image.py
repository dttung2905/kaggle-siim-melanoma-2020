import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
from joblib import Parallel, delayed
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source2019", type=str, required=True, help="source data 2019"
    )
    parser.add_argument(
        "--source2020", type=str, required=True, help="source data 2020"
    )
    parser.add_argument(
        "--inputimgfolder", type=str, required=True, help="image folder input"
    )
    parser.add_argument(
        "--outputimgfolder", type=str, required=True, help="image folder output"
    )
    return parser.parse_args()


def convert(img_file, out_dir):
    out_file = out_dir + "/" + os.path.basename(img_file).replace("jpg", "png")
    image = Image.open(img_file)
    image.resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_file, "png")


if __name__ == "__main__":
    args = get_args()
    train_2020 = pd.read_csv(args.source2020)
    train_2019 = pd.read_csv(args.source2019)

    train_join = pd.merge(
        train_2020, train_2019, how="inner", left_on="image_name", right_on="image"
    )
    train_2019_malignant = train_2019[train_2019["MEL"] == 1]
    train_2019_malignant.shape

    train_malignant_list = train_2019_malignant["image"].values.tolist()

    """
    @TODO : remove this line once we are done with the project
    in_dir = "../input/2019_data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"
    out_dir = "../input/2019_data/isic_2019_v2"
    """
    IMAGE_SIZE = 224

    from PIL import Image, ImageChops

    JPEG_FILES = [
        args.inputimgfolder + picture + ".jpg" for picture in train_malignant_list
    ]

    try:
        os.mkdir(args.outputimgfolder)
    except:
        print("already exist")

    Parallel(n_jobs=32, verbose=10)(
        delayed(convert)(f, args.outputimgfolder) for f in JPEG_FILES
    )
