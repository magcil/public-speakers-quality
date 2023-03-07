import os
import argparse
import numpy as np
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, nargs="?", required=True, 
        help="Input annotation file")

#    parser.add_argument(
#        "-o", "--output", type=str, nargs="?", required=True, 
#        help="Output folder where features are stored")

    flags = parser.parse_args()
    ann_file = flags.input
 
    with open(ann_file, 'r') as fp:
        d = json.load(fp)
    
    
    flow_female = np.array([f['flow']['mean'] for f in d if f['metadata']['Presenter:gender']=='Female'])
    flow_male = np.array([f['flow']['mean'] for f in d if f['metadata']['Presenter:gender']=='Male'])
    flow_female = [i for i in flow_female if i is not None]
    flow_male = [i for i in flow_male if i is not None]
    print(flow_male)
    print(f'female flow average: {np.mean(flow_female)}')
    print(f'male flow average: {np.mean(flow_male)}')

    from scipy.stats import ttest_ind

    t_statistic, p_value = ttest_ind(flow_female, flow_male)
    print(t_statistic, p_value)
    if p_value < 0.05 and t_statistic > 0:
        print("flow_female is significantly larger than flow_male")
    else:
        print("flow_female is not significantly larger than flow_male")
