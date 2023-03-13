import argparse
import inference
import glob
import os
import pickle5 as pickle

def inference_dir(dir, model, ground_truth):
    '''
    :param dir: input DIR 
    :param model: model path
    :ground_truth: ground truth json file
    :return: average prediction
    '''
    model_dict = pickle.load(open(model, 'rb'))
    regressor = model_dict['model']
    mid_window = model_dict['mid_window']
    step = model_dict['mid_step']
    scaler = model_dict['scaler']


    files = glob.glob(os.path.join(dir, "*.wav"))

    for f in files:
        pred = inference.inference(f, model)
        print(f"y={pred:.2f}")


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
    inference_dir(input, model, "")

