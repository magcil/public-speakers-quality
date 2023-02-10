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
    # short term audio features (e.g. mfccs) using pyAudioAnalysis
    fs, x = aIO.read_audio_file(file)
    fv, f_names = sF.feature_extraction(x, fs, window_length * fs,
                                        hop_length * fs)
    return fv, f_names


def melgram_features(file):
    # melgram features using librosa:
    x, fs = librosa.load(file, sr=None)

    # compute mel spectrogram:
    spectrogram = librosa.feature.melspectrogram(y=x, sr=fs,
                                                 n_fft=int(window_length * fs), 
                                                 hop_length=int(hop_length * fs))
    spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_dB


# TODO: add wav2vec2 embeddings here

def feature_extraction_folder(folder_path, out_dir):
    files = glob.glob(os.path.join(folder_path, '*.wav'))
    for f in files:
        fv, f_n = pyaudioanalysis_features(f)
        mel = melgram_features(f)

        print(fv.shape)
        print(mel.shape)

        file_prefix = os.path.join(out_dir, 
                                   os.path.basename(f).replace(".wav", ""))
        print("saving " + file_prefix)
        np.save(file_prefix + "_pyaudioanalysis.npy", fv)
        np.save(file_prefix + "_melgram.npy", mel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, nargs="?", required=True, 
        help="Input folder")

    parser.add_argument(
        "-o", "--output", type=str, nargs="?", required=True, 
        help="Output folder where features are stored")

    flags = parser.parse_args()
    in_dir = flags.input
    out_dir = flags.output

    if not os.path.exists(out_dir):     
        os.makedirs(out_dir)
    feature_extraction_folder(in_dir, out_dir)