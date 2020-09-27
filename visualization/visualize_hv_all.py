# plot comparison of hypervolume over all runs for all algorithms on all problems

import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 15
MAX_SIZE=18
plt.rc('font', family='Times New Roman', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MAX_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--n-seed', type=int, default=10, help='number of different seeds')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--savefig', default=False, action='store_true', help='saving instead of showing the plot')
    parser.add_argument('--num-eval', type=int, default=200, help='number of evaluations')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    problems = ['zdt1', 'zdt2', 'zdt3', 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'oka1', 'oka2', 'vlmop2', 'vlmop3',
        're1', 're2', 're3', 're4', 're5', 're6', 're7']
    algos = {'nsga2': 'NSGA-II', 'parego': 'ParEGO', 'moead-ego': 'MOEA/D-EGO', 'tsemo': 'TSEMO', 'usemo-ei': 'USeMO-EI', 'dgemo': 'DGEMO (Ours)'}

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')

    n_row, n_col = 4, 5
    fig, axes = plt.subplots(n_row, n_col, figsize=(18, 12))
    n_algo, n_seed, num_eval = len(algos), args.n_seed, args.num_eval
    colors = ['tomato', 'slategray', 'dodgerblue', 'orange', 'mediumaquamarine', 'mediumslateblue']
    
    for pid, problem in enumerate(problems):
        problem_dir = os.path.join(result_dir, problem, args.subfolder)

        # read result csvs
        data_list = [[] for _ in range(n_algo)]
        for i, algo in enumerate(algos.keys()):
            for seed in range(n_seed):
                csv_path = f'{problem_dir}/{algo}/{seed}/EvaluatedSamples.csv'
                data_list[i].append(pd.read_csv(csv_path))

        # get statistics
        num_init_samples = sum(data_list[0][0]['iterID'] == 0)
        batch_size = sum(data_list[0][0]['iterID'] == 1)

        num_samples = num_init_samples + num_eval
        sample_ids = np.arange(num_eval + 1)

        # calculate hypervolume
        hv = np.zeros((n_algo, n_seed, num_eval + 1))
        for i in range(n_algo):
            for j in range(n_seed):
                hv_data = data_list[i][j]['Hypervolume_indicator']
                hv[i][j] = np.concatenate([np.full(batch_size, hv_data[0]), hv_data[num_init_samples:num_samples - batch_size + 1]])

        row_id, col_id = pid // n_col, pid % n_col
        ax = axes[row_id, col_id]
        ax.set_title(problem.upper())
        for i in range(n_algo):
            hv_mean, hv_std = hv[i].mean(axis=0), hv[i].std(axis=0)
            ax.plot(sample_ids, hv_mean, color=colors[i], label=list(algos.values())[i] if pid == 0 else '')
            ax.fill_between(sample_ids, hv_mean - 0.5 * hv_std, hv_mean + 0.5 * hv_std, color=colors[i], alpha=0.1)
        ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        ax.locator_params(nbins=5, axis='x')

        if row_id == n_row - 1:
            ax.set_xlabel('Evaluations')
        if col_id == 0:
            ax.set_ylabel('Hypervolume')

    fig.legend(loc='lower center', ncol=n_algo, fontsize='large')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10, left=0.06)

    if args.savefig:
        plt.savefig('final_hv.png', bbox_inches='tight')
    else:
        plt.show()
        

if __name__ == '__main__':
    main()
