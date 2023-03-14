import argparse
import inference
import glob
import os
import json
import numpy as np


def inference_dir(dir, model, ground_truth_file):
    '''
    :param dir: input DIR 
    :param model: model path
    :ground_truth: ground truth json file
    :return: average prediction
    '''
    files = glob.glob(os.path.join(dir, "*.wav"))

    with open(ground_truth_file) as fp:
        ground_truth = json.load(fp)
    errors = []
    for f in files[::40]:
        f2 = os.path.basename(f)
        found = False
        for g in ground_truth:
            if g['name'] == f2:
                found = True
                break
        if found:
            model_name = model.split("/")[-1].split('.')[0]
            real = g[model_name]['mean']
            pred = inference.inference(f, model)
            errors.append(np.abs(real-pred))
            print(f"pred={pred:.2f} real={real:.2f}")
        print(np.mean(errors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_audio", type=str, nargs="?", required=True,
        help="input audio directory")

    parser.add_argument("-m", "--model_path", required=False,
                        help="the trained model to be used")


    flags = parser.parse_args()
    input = flags.input_audio
    model = flags.model_path
    inference_dir(input, model, "annotations_metadata.json")

