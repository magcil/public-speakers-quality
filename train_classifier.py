
import argparse
import glob
import numpy as np
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--ground_truth", type=str, nargs="?", required=True, 
        help="Ground truth and metadata json file")

    parser.add_argument(
        "-f", "--features_folder", type=str, nargs="?", required=True, 
        help="Folder where the features are stored in npy files")

    flags = parser.parse_args()
    gt_file = flags.ground_truth
    f_dir = flags.features_folder

    with open(gt_file) as fp:
        ground_truth = json.load(fp)

    print(ground_truth[0])