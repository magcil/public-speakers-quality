import librosa
import os
import argparse
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures as sF
import glob
import numpy as np


window_length = (40 * 1e-3)
hop_length = (20 * 1e-3)


def pyaudioanalysis_features(file):
    fs, x = aIO.read_audio_file(file)
    fv, f_names = sF.feature_extraction(x, fs, window_length * fs,
                                        hop_length * fs)
    return fv, f_names


def melgram_features(file):
    x, fs = librosa.load(file, sr=None)

    # compute mel spectrogram:
    spectrogram = librosa.feature.melspectrogram(y=x, sr=fs,
                                                 n_fft=int(window_length * fs), 
                                                 hop_length=int(hop_length * fs))
    spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_dB

def feature_extraction_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.wav'))
    print(files)
    for f in files:
        fv, f_n = pyaudioanalysis_features(f)
        mel = melgram_features(f)

        print(fv.shape)
        print(mel.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, nargs="?", required=True, 
        help="Input folder"
    )

    parser.add_argument(
        "-o", "--output", type=str, nargs="?", required=True, 
        help="Output folder where features are stored"
    )

    flags = parser.parse_args()
    csv_in = flags.input
    out_dir = flags.output

    if not os.path.exists(out_dir):     
        os.makedirs(out_dir)
    feature_extraction_folder(csv_in)