## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

## Download training and testing data for image deraining task
import os
import gdown
import shutil

import argparse
from typing import Literal


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    choices=["msrb", "none"],
    type=str,
    required=True,
    help="msrb or none",
)
args = parser.parse_args()

msrb_dataset = "1OMNWB-5c3CoKrLgTAzcNAZj6gA4jxIYU"

for data in args.data.split("-"):
    data = data.lower()
    if data == "msrb":
        print("Download MSRB dataset")
        gdown.download(id=msrb_dataset, output="Datasets/msrb.zip", quiet=False)
        print("Extracting MSRB data...")
        shutil.unpack_archive("Datasets/msrb.zip", "Datasets")
        os.remove("Datasets/msrb.zip")
