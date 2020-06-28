import os, sys
os.environ['OMP_NUM_THREADS'] = '1' # speed up
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from argparse import ArgumentParser
import numpy as np
from time import time
import yaml
from multiprocessing import cpu_count
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from problems.common import build_problem
from external import lhs
from data_export import DataExport
from mobo.utils import calc_hypervolume
from utils import get_result_dir, setup_logger


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default='dtlz1', 
        help='optimization problem')
    parser.add_argument('--n-var', type=int, default=6, 
        help='number of design variables')
    parser.add_argument('--n-obj', type=int, default=2, 
        help='number of objectives')
    parser.add_argument('--n-init-sample', type=int, default=50, 
        help='number of initial design samples')
    parser.add_argument('--n-iter', type=int, default=20, 
        help='number of optimization iterations')
    parser.add_argument('--ref-point', type=float, nargs='+', default=None, 
        help='reference point for calculating hypervolume')
    parser.add_argument('--batch-size', type=int, default=20, 
        help='size of the evaluated batch in one iteration')
    parser.add_argument('--pop-init-method', type=str, choices=['nds', 'random', 'lhs'], default='nds', 
        help='method to init population')
        
    parser.add_argument('--seed', type=int, default=0, 
        help='random seed')
    parser.add_argument('--subfolder', type=str, default='default',
        help='subfolder name for storing results, directly store under result/ as default')
    parser.add_argument('--exp-name', type=str, default=None,
        help='custom experiment name to distinguish between experiments on same problem and same algorithm')
    parser.add_argument('--log-to-file', default=False, action='store_true',
        help='log output to file rather than print by stdout')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')

    args = parser.parse_args()
    return args


def get_ref_point(problem, n_var=6, n_obj=2, n_init_sample=50, seed=0):

    np.random.seed(seed)
    _, _, _, Y_init = build_problem(problem, n_var, n_obj, n_init_sample, n_process=1)

    return np.max(Y_init, axis=0)


def save_args(args):
    '''
    Save arguments to yaml file
    '''
    result_dir = get_result_dir(args)
    args_path = os.path.join(result_dir, 'args.yml')
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False, sort_keys=False)


def main():
    # get argument values
    args = get_args()

    # get reference point
    if args.ref_point is None:
        args.ref_point = get_ref_point(args.problem, args.n_var, args.n_obj, args.n_init_sample)

    t0 = time()

    # set seed
    np.random.seed(args.seed)

    # build problem, get initial samples
    problem, true_pfront, X_init, Y_init = build_problem(args.problem, args.n_var, args.n_obj, args.n_init_sample, args.n_process)
    args.n_var, args.n_obj, args.algo = problem.n_var, problem.n_obj, 'nsga2'

    # save arguments and setup logger
    save_args(args)
    logger = setup_logger(args)
    print(problem)

    # initialize data exporter
    exporter = DataExport(X_init, Y_init, args)

    # initialize population
    if args.pop_init_method == 'lhs':
        sampling = LatinHypercubeSampling()
    elif args.pop_init_method == 'nds':
        sorted_indices = NonDominatedSorting().do(Y_init)
        sampling = X_init[np.concatenate(sorted_indices)][:args.batch_size]
        if len(sampling) < args.batch_size:
            rest_sampling = lhs(X_init.shape[1], args.batch_size - len(sampling))
            sampling = np.vstack([sampling, rest_sampling])
    elif args.pop_init_method == 'random':
        sampling = FloatRandomSampling()
    else:
        raise NotImplementedError

    # initialize evolutionary algorithm
    ea_algorithm = NSGA2(pop_size=args.batch_size, sampling=sampling)

    # find Pareto front
    res = minimize(problem, ea_algorithm, ('n_gen', args.n_iter), save_history=True)
    X_history = np.array([algo.pop.get('X') for algo in res.history])
    Y_history = np.array([algo.pop.get('F') for algo in res.history])

    # update data exporter
    for X_next, Y_next in zip(X_history, Y_history):
        exporter.update(X_next, Y_next)

    # export all result to csv
    exporter.write_csvs()
    if true_pfront is not None:
        exporter.write_truefront_csv(true_pfront)

    # statistics
    final_hv = calc_hypervolume(exporter.Y, exporter.ref_point)
    print('========== Result ==========')
    print('Total runtime: %.2fs' % (time() - t0))
    print('Total evaluations: %d, hypervolume: %.4f\n' % (args.batch_size * args.n_iter, final_hv))

    # close logger
    if logger is not None:
        logger.close()


if __name__ == '__main__':
    main()