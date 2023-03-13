import argparse
from feature_extraction import pyaudioanalysis_features,mid_feature_extraction
import pickle5 as pickle
import numpy as np

def inference(input, model):
    '''

    :param input:
    :param model:
    :return:
    '''
    model_dict = pickle.load(open(model, 'rb'))
    regressor = model_dict['model']
    mid_window = model_dict['mid_window']
    step = model_dict['mid_step']
    scaler = model_dict['scaler']
    fv, f_n = pyaudioanalysis_features(input)
    mtf = mid_feature_extraction(fv, mid_window, step, 0.04, 0.02)
    X = []
    for segment in mtf.T:
        X.append(np.array(segment))
    X = scaler.transform(X)
    preds = regressor.predict(X)
    average_prediction = round(np.average(preds), 2)
    model_name = model.split("/")[-1].split('.')[0]
    print("Predicted value for task ", model_dict, " is: ", average_prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_audio", type=str, nargs="?", required=True,
        help="input audio wav file")

    parser.add_argument("-m", "--model_path", required=False,
                        help="the trained model to be used")


    flags = parser.parse_args()
    input = flags.input_audio
    model = flags.model_path
    inference(input, model)

