# plot comparison of hypervolume indicator over all runs for any algorithms

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

    n_algo, n_seed, seed = len(algo_names), args.n_seed, args.seed

    # read result csvs
    # calculate average hypervolume indicator across seeds
    ds = [{} for _ in range(n_algo)]
    data_list = [[] for _ in range(n_algo)]
    avgHV = np.zeros(n_algo)
    avgHV_all = [None for _ in range(n_algo)]
    num_init_samples = None
    batch_size = None
    for i in range(n_algo):
        for j in range(n_seed):
            if n_seed == 1: j = seed
            csv_path = f'{problem_dir}/{algo_names[i]}/{j}/EvaluatedSamples.csv'
            df = pd.read_csv(csv_path)
            data_list[i].append(df)
            if num_init_samples is None and 'iterID' in df:
                num_init_samples = sum(df['iterID'] == 0)
                batch_size = sum(df['iterID'] == 1)
            ds[i][f'Hypervolume_indicator_{j + 1}'] = df['Hypervolume_indicator']
            avgHV[i] += df['Hypervolume_indicator'][df.index[-1]] / n_seed
            if avgHV_all[i] is None:
                avgHV_all[i] = df['Hypervolume_indicator'] / n_seed
            else:
                avgHV_all[i] += df['Hypervolume_indicator'] / n_seed
    
    df_HV_list = [pd.DataFrame(d) for d in ds]

    sampleIds = list(range(1, len(df_HV_list[0]) + 1))
    assert num_init_samples is not None

    # dataframe for boxplot of Hypervolume indicator
    boxplot_col_names = ['SampleId', 'Hypervolume_indicator', 'RunId', 'AlgorithmId']
    df_boxplot = pd.DataFrame(columns=boxplot_col_names)
    for i in range(n_algo):
        for j in range(n_seed):
            if n_seed == 1: j = seed
            num_alg_eval = df_HV_list[i].shape[0] - num_init_samples
            for k in [0.25 * num_alg_eval, 0.5 * num_alg_eval, 0.75 * num_alg_eval, num_alg_eval]:
                df_boxplot = df_boxplot.append({
                    'SampleId': k, 
                    'Hypervolume_indicator': df_HV_list[i][f'Hypervolume_indicator_{j + 1}'][k + num_init_samples - 1],
                    'RunId': j,
                    'AlgorithmId': i
                    }, ignore_index=True)

    # calculate proper range of plot
    minHV = min([min(df.min()) for df in df_HV_list])
    maxHV = max([max(df.max()) for df in df_HV_list])
    minAvgHV = np.min(avgHV_all)
    maxAvgHV = np.max(avgHV_all)

    for i in range(n_algo):
        print(f'{args.problem}: {algo_names[i]} average HV indicator after {num_alg_eval} evaluations: {avgHV[i]}')

    # Starting the Figure
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=[f'Hypervolume Indicator over {n_seed} runs'] * 2,
                        specs=[[{"type": "scatter"}], [{"type": "box"}]],
                        vertical_spacing=0.1)

    # Plotting Hypervolume Indicator
    for i in range(n_algo):
        fig.add_trace(go.Scatter(
            name=algo_names[i], 
            x=sampleIds, y=avgHV_all[i], 
            mode='lines', line=dict(color=defaultColors[i])),
            row=1,col=1)
    fig.update_xaxes(title_text='# Samples', row=1, col=1)
    fig.update_yaxes(title_text='Hypervolume Indicator', range=[minAvgHV - 0.05 * (maxAvgHV - minAvgHV), maxAvgHV + 0.05 * (maxAvgHV - minAvgHV)], row=1, col=1)

    # adding box plot
    for i in range(n_algo):
        algBox = df_boxplot[df_boxplot['AlgorithmId'] == i]
        fig.add_trace(go.Box(
            y=algBox['Hypervolume_indicator'], x=algBox['SampleId'], 
            name=algo_names[i], marker_color=defaultColors[i]),
            row=2, col=1)
    fig.update_xaxes(title_text='# Evaluations', row=2, col=1)
    fig.update_yaxes(title_text='Hypervolume Indicator', range=[minHV - 0.05 * (maxHV - minHV), maxHV + 0.05 * (maxHV - minHV)], row=2, col=1)

    # Changing Overall Size
    fig.update_layout(
        width=1200,
        height=1200,
        boxmode='group', # group together boxes of the different traces for each value of x
        title=f'Hypervolume Indicator Over {n_seed} Runs for {args.problem}, {num_init_samples} initial samples, batch size {batch_size}'
    )

    if args.savefig:
        fig.write_image(f'{args.problem}_hv.png')
    else:
        fig.show()


if __name__ == '__main__':
    main()
