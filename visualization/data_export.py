import os
import pandas as pd
import numpy as np
from mobo.utils import find_pareto_front, calc_hypervolume
from utils import get_result_dir

'''
Export csv files for external visualization.
'''

class DataExport:

    def __init__(self, optimizer, X, Y, args):
        '''
        Initialize data exporter from initial data (X, Y).
        '''
        self.optimizer = optimizer
        self.problem = optimizer.real_problem
        self.n_var, self.n_obj = self.problem.n_var, self.problem.n_obj
        self.batch_size = self.optimizer.selection.batch_size
        self.iter = 0
        self.transformation = optimizer.transformation

        # saving path related
        self.result_dir = get_result_dir(args)
        
        n_samples = X.shape[0]

        # compute initial hypervolume
        pfront, pidx = find_pareto_front(Y, return_index=True)
        pset = X[pidx]
        if args.ref_point is None:
            args.ref_point = np.max(Y, axis=0)
        hv_value = calc_hypervolume(pfront, ref_point=args.ref_point)
        
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

        # predicted performance
        for i in range(self.n_obj):
            obj_pred_name = f'Expected_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f'Uncertainty_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f'Acquisition_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)

        d1['Hypervolume_indicator'] = np.full(n_samples, hv_value)

        self.export_data = pd.DataFrame(data=d1) # export all data
        self.export_pareto = pd.DataFrame(data=d2) # export pareto data
        column_names.append('ParetoFamily')
        self.export_approx_pareto = pd.DataFrame(columns=column_names) # export pareto approximation data

        self.has_family = hasattr(self.optimizer.selection, 'has_family') and self.optimizer.selection.has_family

    def update(self, X_next, Y_next):
        '''
        For each algorithm iteration adds data for visualization.
        Input:
            X_next: proposed sample values in design space
            Y_next: proposed sample values in performance space
        '''
        self.iter += 1

        # evaluate prediction of X_next on surrogate model
        val = self.optimizer.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True)
        Y_next_pred_mean = self.transformation.undo(y=val['F'])
        Y_next_pred_std = val['S']
        acquisition, _, _ = self.optimizer.acquisition.evaluate(val)

        pset = self.optimizer.status['pset']
        pfront = self.optimizer.status['pfront']
        hv_value = self.optimizer.status['hv']

        d1 = {'iterID': np.full(self.batch_size, self.iter, dtype=int)} # export all data
        d2 = {'iterID': np.full(pfront.shape[0], self.iter, dtype=int)} # export pareto data

        # design variables
        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X_next[:, i]
            d2[var_name] = pset[:, i]

        # performance and predicted performance
        for i in range(self.n_obj):
            col_name = f'f{i + 1}'
            d1[col_name] = Y_next[:, i]
            d2['Pareto_'+col_name] = pfront[:, i]

            col_name = f'Expected_f{i + 1}'
            d1[col_name] = Y_next_pred_mean[:, i]
            col_name = f'Uncertainty_f{i + 1}'
            d1[col_name] = Y_next_pred_std[:, i]
            col_name = f'Acquisition_f{i + 1}'
            d1[col_name] = acquisition[:, i]

        d1['Hypervolume_indicator'] = np.full(self.batch_size, hv_value)

        if self.has_family:
            info = self.optimizer.info
            family_lbls, approx_pset, approx_pfront = info['family_lbls'], info['approx_pset'], info['approx_pfront']
            approx_front_samples = approx_pfront.shape[0]
            
            d3 = {'iterID': np.full(approx_front_samples, self.iter, dtype=int)} # export pareto approximation data

            for i in range(self.n_var):
                var_name = f'x{i + 1}'
                d3[var_name] = approx_pset[:, i]

            for i in range(self.n_obj):
                d3[f'Pareto_f{i + 1}'] = approx_pfront[:, i]

            d3['ParetoFamily'] = family_lbls
        
        else:
            approx_pset = self.optimizer.solver.solution['x']
            val = self.optimizer.surrogate_model.evaluate(approx_pset)
            approx_pfront = val['F']
            approx_pset, approx_pfront = self.transformation.undo(approx_pset, approx_pfront)

            # find undominated
            approx_pfront, pidx = find_pareto_front(approx_pfront, return_index=True)
            approx_pset = approx_pset[pidx]
            approx_front_samples = approx_pfront.shape[0]

            d3 = {'iterID': np.full(approx_front_samples, self.iter, dtype=int)}

            for i in range(self.n_var):
                var_name = f'x{i + 1}'
                d3[var_name] = approx_pset[:, i]

            for i in range(self.n_obj):
                d3[f'Pareto_f{i + 1}'] = approx_pfront[:, i]

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
