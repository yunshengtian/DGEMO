# plot comparison of expected hypervolume improvement, prediction error, 
# and hypervolume indicator of chosen test run for two algorithms 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColors


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)

    n_algo = len(algo_names)

    # read result csvs
    data_list = []
    for algo_name in algo_names:
        csv_path = f'{problem_dir}/{algo_name}/{args.seed}/EvaluatedSamples.csv'
        data_list.append(pd.read_csv(csv_path))

    sampleIds = list(range(1, len(data_list[0]) + 1))
    num_init_samples = sum(data_list[0]['iterID'] == 0)
    batch_size = sum(data_list[0]['iterID'] == 1)
    num_eval = len(sampleIds) - num_init_samples
    n_obj = len([key for key in data_list[0] if key.startswith('f')])

    # calculate hypervolume and average prediction error
    pred_error_list = [[None for _ in range(n_obj)] for _ in range(n_algo)]
    hv_list = [None for _ in range(n_algo)]

    for i in range(n_algo):
        for j in range(n_obj):
            pred_error_list[i][j] = abs(data_list[i][f'f{j + 1}'] - data_list[i][f'Expected_f{j + 1}'])[num_init_samples:]
            pred_error_list[i][j] = np.cumsum(pred_error_list[i][j]) / np.arange(num_eval)
        hv_list[i] = data_list[i]['Hypervolume_indicator']
    
    # calculate proper range of plot
    minErr, maxErr = np.min(pred_error_list), np.max(pred_error_list)
    minHV, maxHV = np.min(hv_list), np.max(hv_list)

    fig = make_subplots(rows=2, cols=n_algo, column_titles=algo_names)

    for i in range(n_algo):
        # Plotting Hypervolume Indicator
        fig.add_trace(go.Scatter(
            name=algo_names[i] + ' Hypervolume Indicator', 
            x=sampleIds, y=data_list[i]['Hypervolume_indicator'], 
            mode='lines', line=dict(color='#AB63FA')),
            row=1, col=i + 1)
        fig.update_xaxes(title_text='# Samples', row=1, col=i + 1)
        fig.update_yaxes(title_text='Hypervolume Indicator', range=[minHV - .05 * (maxHV - minHV), maxHV + .05 * (maxHV - minHV)], row=1, col=i + 1)

        # Plotting Prediction Error
        for j in range(n_obj):
            fig.add_trace(go.Scatter(
                name=algo_names[i] + ' f' + str(j + 1), 
                x=sampleIds[num_init_samples:], y=pred_error_list[i][j], 
                mode='lines', line=dict(color=defaultColors[j])), 
                row=2, col=i + 1)
        fig.update_xaxes(title_text='# Evaluations', row=2, col=i + 1)
        fig.update_yaxes(title_text='Average Prediction Error', range=[minErr - .05 * (maxErr - minErr), maxErr + .05 * (maxErr - minErr)], row=2, col=i + 1)

    # Changing Overall Size
    fig.update_layout(
        width=1000 + 300 * n_algo,
        height=1200,
        title=f'Algorithm Performance Comparison for {args.problem}, batch size {batch_size}, test run {args.seed}'
    )

    fig.show()


if __name__ == '__main__':
    main()