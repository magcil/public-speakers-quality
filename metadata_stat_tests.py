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

    parser.add_argument(
        "-t", "--task", type=str, nargs=None, required=True,
        choices=['confidence', 'fillers', 'flow', 'overall', 'intonation'],
        help="Task")

    flags = parser.parse_args()
    ann_file = flags.input
    task = flags.task
 
    with open(ann_file, 'r') as fp:
        d = json.load(fp)
    
    female = np.array([f[task]['mean'] for f in d if f['metadata']['Presenter:gender']=='Female']) 
    male = np.array([f[task]['mean'] for f in d if f['metadata']['Presenter:gender']=='Male'])
    female = [i for i in female if i is not None]
    male = [i for i in male if i is not None]
    print(f'female {task} average: {np.mean(female)}')
    print(f'male {task} average: {np.mean(male)}')

    from scipy.stats import ttest_ind

    t_statistic, p_value = ttest_ind(female, male)
    print(f't_stat={t_statistic:.5f} - p_value={p_value:.5f}')
    if p_value < 0.05 and t_statistic > 0:
        print("female is significantly larger than male")
    elif p_value < 0.05 and t_statistic <= 0:
        print("female is significantly smaller than male")
    else:
        print("female is not significantly larger than male")
