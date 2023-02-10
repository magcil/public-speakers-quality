import argparse
import json
import pandas as pd
import numpy as  np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def data_parser(metadata, annotations):
    f = open(annotations)
    data = json.load(f)
    df = pd.read_csv(metadata)
    samples = [] #list of dictionaries
    for sample in data:
        url = "_".join(sample["name"].split("_")[4:])
        url = url.split(".wav")[0]
        contain_values = df.loc[df['Youtube_link'].str.contains(url)]
        sample["metadata"] = {}
        for col in contain_values.columns:
             sample["metadata"][col] = contain_values.iloc[0][col]
        samples.append(sample)
    with open('annotations_metadata.json', 'w') as fout:
        json.dump(samples, fout, cls=NpEncoder, indent = 1)


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metadata', required=True,
                        type=str, help='metadata csv')
    parser.add_argument('-a', '--annotations', required=True,
                        type=str, help='annotations json')
    args = parser.parse_args()
    data_parser(args.metadata, args.annotations)













