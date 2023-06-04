import os
import argparse
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kruskal


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
    print(f'female {task} average: {np.mean(female):.2f}')
    print(f'male {task} average: {np.mean(male):.2f}')
    print(f'female {task} std: {np.std(female):.2f}')
    print(f'male {task} std: {np.std(male):.2f}')

    s_senior = np.array([f[task]['mean'] for f in d if f['metadata']['Presenter:seniority']=='Senior Academic'])
    s_junior = np.array([f[task]['mean'] for f in d if f['metadata']['Presenter:seniority']=='Junior Academic'])
    s_senior = [i for i in s_senior if i is not None]
    s_junior = [i for i in s_junior if i is not None]
    print(f's_senior {task} average: {np.mean(s_senior):.2f}')
    print(f's_junior {task} average: {np.mean(s_junior):.2f}')
    print(f's_senior {task} std: {np.std(s_senior):.2f}')
    print(f's_junior {task} std: {np.std(s_junior):.2f}')

    t_statistic, p_value = kruskal(female, male)
    print(f't_stat={t_statistic:.5f} - p_value={p_value:.5f}')
    if p_value < 0.05 and t_statistic > 0:
        print("female is significantly larger than male")
    elif p_value < 0.05 and t_statistic <= 0:
        print("female is significantly smaller than male")
    else:
        print("female is not significantly larger than male")

    t_statistic, p_value = kruskal(s_senior, s_junior)
    print(f't_stat={t_statistic:.5f} - p_value={p_value:.5f}')
    if p_value < 0.05 and t_statistic > 0:
        print("s_senior is significantly larger than s_junior")
    elif p_value < 0.05 and t_statistic <= 0:
        print("s_senior is significantly smaller than s_junior")
    else:
        print("s_senior is not significantly larger than s_junior")


    plot_hists = True
    if plot_hists:
        h1, v1 = np.histogram(female)
        h2, v2 = np.histogram(male)
        v1 = (v1[0:-1] + v1[1:]) / 2
        v2 = (v2[0:-1] + v2[1:]) / 2
        plt.subplot(2,1,1)
        plt.plot(v1, h1)
        plt.plot(v2, h2)
        plt.title('histograms of male vs female')
        h1, v1 = np.histogram(s_senior)
        h2, v2 = np.histogram(s_junior)
        v1 = (v1[0:-1] + v1[1:]) / 2
        v2 = (v2[0:-1] + v2[1:]) / 2
        plt.subplot(2,1,2)
        plt.plot(v1, h1)
        plt.plot(v2, h2)
        plt.title('histograms of senior vs junior')
        plt.show()
