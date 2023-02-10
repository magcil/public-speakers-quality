import librosa
import os
import argparse
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures as sF
import glob

window_length = (40 * 1e-3)
hop_length = (20 * 1e-3)


def pyaudioanalysis_features(file):
    fs, x = aIO.read_audio_file(file)
    fv, f_names = sF.feature_extraction(x, fs, window_length * fs,
                                        hop_length * fs)
    return fv, f_names


def feature_extraction_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.wav'))
    print(files)
    for f in files:
        f, f_n = pyaudioanalysis_features(f)
        print(f.shape)

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