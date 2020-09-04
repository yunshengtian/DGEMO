import os
import pandas as pd
import numpy as np
from mobo.utils import find_pareto_front, calc_hypervolume
from utils import get_result_dir

'''
Export csv files for external visualization (for moo only).
'''

class DataExport:

    def __init__(self, X, Y, args):
        '''
        Initialize data exporter from initial data (X, Y).
        '''
        self.n_var, self.n_obj = args.n_var, args.n_obj
        self.batch_size = args.batch_size
        self.iter = 0
        self.X, self.Y = X, Y
        self.ref_point = np.max(Y, axis=0) if args.ref_point is None else args.ref_point

        # saving path related
        self.result_dir = get_result_dir(args)
        
        n_samples = X.shape[0]

        # compute hypervolume
        pfront, pidx = find_pareto_front(Y, return_index=True)
        pset = X[pidx]
        hv_value = calc_hypervolume(pfront, ref_point=self.ref_point)
        
        # init data frame
        column_names = ['iterID']
        d1 = {'iterID': np.zeros(n_samples, dtype=int)}
        d2 = {'iterID': np.zeros(len(pset), dtype=int)}

        # design variables
        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X[:, i]
            d2[var_name] = pset[:, i]
            column_names.append(var_name)

        # performance
        for i in range(self.n_obj):
            obj_name = f'f{i + 1}'
            d1[obj_name] = Y[:, i]
            obj_name = f'Pareto_f{i + 1}'
            d2[obj_name] = pfront[:, i]

        d1['Hypervolume_indicator'] = np.full(n_samples, hv_value)

        self.export_data = pd.DataFrame(data=d1) # export all data
        self.export_pareto = pd.DataFrame(data=d2) # export pareto data
        column_names.append('ParetoFamily')
        self.export_approx_pareto = pd.DataFrame(columns=column_names) # export pareto approximation data

    def update(self, X_next, Y_next):
        '''
        For each algorithm iteration adds data for visualization.
        Input:
            X_next: proposed sample values in design space
            Y_next: proposed sample values in performance space
        '''
        self.iter += 1
        self.X = np.vstack([self.X, X_next])
        self.Y = np.vstack([self.Y, Y_next])

        # compute hypervolume
        pfront, pidx = find_pareto_front(self.Y, return_index=True)
        pset = self.X[pidx]
        hv_value = calc_hypervolume(pfront, ref_point=self.ref_point)

        approx_pfront, pidx = find_pareto_front(Y_next, return_index=True)
        approx_pset = X_next[pidx]
        approx_front_samples = approx_pfront.shape[0]

        d1 = {'iterID': np.full(self.batch_size, self.iter, dtype=int)} # export all data
        d2 = {'iterID': np.full(pfront.shape[0], self.iter, dtype=int)} # export pareto data
        d3 = {'iterID': np.full(approx_front_samples, self.iter, dtype=int)} # export pareto approximation data

        # design variables
        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X_next[:, i]
            d2[var_name] = pset[:, i]
            d3[var_name] = approx_pset[:, i]

        # performance and predicted performance
        for i in range(self.n_obj):
            col_name = f'f{i + 1}'
            d1[col_name] = Y_next[:, i]
            d2['Pareto_'+col_name] = pfront[:, i]
            d3['Pareto_'+col_name] = approx_pfront[:, i]

        d1['Hypervolume_indicator'] = np.full(self.batch_size, hv_value)
        d3['ParetoFamily'] = np.zeros(approx_front_samples)

        df1 = pd.DataFrame(data=d1)
        df2 = pd.DataFrame(data=d2)
        df3 = pd.DataFrame(data=d3)
        self.export_data = self.export_data.append(df1, ignore_index=True)
        self.export_pareto = self.export_pareto.append(df2, ignore_index=True)
        self.export_approx_pareto = self.export_approx_pareto.append(df3, ignore_index=True)

    def write_csvs(self):
        '''
        Export data to csv files.
        '''
        dataframes = [self.export_data, self.export_pareto, self.export_approx_pareto]
        filenames = ['EvaluatedSamples', 'ParetoFrontEvaluated','ParetoFrontApproximation']

        for dataframe, filename in zip(dataframes, filenames):
            filepath = os.path.join(self.result_dir, filename + '.csv')
            dataframe.to_csv(filepath, index=False)

    def write_truefront_csv(self, truefront):
        '''
        Export true pareto front to csv files.
        '''
        problem_dir = os.path.join(self.result_dir, '..', '..') # result/problem/subfolder/
        filepath = os.path.join(problem_dir, 'TrueParetoFront.csv')

        if os.path.exists(filepath): return

        d = {}
        for i in range(truefront.shape[1]):
            col_name = f'f{i + 1}'
            d[col_name] = truefront[:, i]

        export_tf = pd.DataFrame(data=d)
        export_tf.to_csv(filepath, index=False)
