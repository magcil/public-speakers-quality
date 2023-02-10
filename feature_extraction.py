import librosa
import os
import argparse

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
