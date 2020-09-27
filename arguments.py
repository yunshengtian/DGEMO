import os
from argparse import ArgumentParser, Namespace
import yaml
from multiprocessing import cpu_count

'''
Get argument values from command line
Here we speficy different argument parsers to avoid argument conflict between initializing each components
'''

def get_general_args(args=None):
    '''
    General arguments: problem and algorithm description, experiment settings
    '''
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
    parser.add_argument('--batch-size', type=int, default=10, 
        help='size of the selected batch in one iteration')

    parser.add_argument('--seed', type=int, default=0, 
        help='random seed')
    parser.add_argument('--n-seed', type=int, default=1,
        help='number of random seeds / test runs')

    parser.add_argument('--algo', type=str, default='dgemo',
        help='type of algorithm to use with some predefined arguments, or custom arguments')

    parser.add_argument('--subfolder', type=str, default='default',
        help='subfolder name for storing results, directly store under result/ as default')
    parser.add_argument('--exp-name', type=str, default=None,
        help='custom experiment name to distinguish between experiments on same problem and same algorithm')
    parser.add_argument('--log-to-file', default=False, action='store_true',
        help='log output to file rather than print by stdout')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')

    args, _ = parser.parse_known_args(args)
    return args


def get_surroagte_args(args=None):
    '''
    Arguments for fitting the surrogate model
    '''
    parser = ArgumentParser()

    parser.add_argument('--surrogate', type=str, 
        choices=['gp', 'ts'], default='gp', 
        help='type of the surrogate model')
    parser.add_argument('--n-spectral-pts', type=int, default=100, 
        help='number of points for spectral sampling')
    parser.add_argument('--nu', type=int,
        choices=[1, 3, 5, -1], default=5,
        help='parameter nu for matern kernel (integer, -1 means inf)')
    parser.add_argument('--mean-sample', default=False, action='store_true', 
        help='use mean sample when sampling objective functions')

    args, _ = parser.parse_known_args(args)
    return args


def get_acquisition_args(args=None):
    '''
    Arguments for acquisition function
    '''
    parser = ArgumentParser()

    parser.add_argument('--acquisition', type=str,  
        choices=['identity', 'pi', 'ei', 'ucb'], default='identity', 
        help='type of the acquisition function')

    args, _ = parser.parse_known_args(args)
    return args


def get_solver_args(args=None):
    '''
    Arguments for multi-objective solver
    '''
    parser = ArgumentParser()

    # general solver
    parser.add_argument('--solver', type=str, 
        choices=['nsga2', 'moead', 'discovery'], default='nsga2', 
        help='type of the multiobjective solver')
    parser.add_argument('--pop-size', type=int, default=100, 
        help='population size')
    parser.add_argument('--n-gen', type=int, default=10, 
        help='number of generations')
    parser.add_argument('--pop-init-method', type=str, 
        choices=['nds', 'random', 'lhs'], default='nds', 
        help='method to init population')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')
    parser.add_argument('--batch-size', type=int, default=20, 
        help='size of the selected batch in one iteration')

    # ParetoDiscovery solver
    parser.add_argument('--n-cell', type=int, default=None,
        help='number of cells in performance buffer, default: 100 for 2-obj, 1000 for 3-obj')
    parser.add_argument('--cell-size', type=int, default=10,
        help='maximum number of samples inside each cell of performance buffer, 0 or negative value means no limit')
    parser.add_argument('--buffer-origin', type=float, nargs='+', default=None,
        help='the origin point of performance buffer, None means 0 as origin')
    parser.add_argument('--buffer-origin-constant', type=float, default=1e-2,
        help='when evaluted value surpasses the buffer origin, adjust the origin accordingly and subtract this constant')
    parser.add_argument('--delta-b', type=float, default=0.2,
        help='unary energy normalization constant for sparse approximation, see section 6.4')
    parser.add_argument('--label-cost', type=int, default=10,
        help='for reducing number of unique labels in sparse approximation, see section 6.4')
    parser.add_argument('--delta-p', type=float, default=10.0,
        help='factor of perturbation in stochastic sampling, see section 6.2.2')
    parser.add_argument('--delta-s', type=float, default=0.3,
        help='scaling factor for choosing reference point in local optimization, see section 6.2.3')
    parser.add_argument('--n-grid-sample', type=int, default=100,
        help='number of samples on local manifold (grid), see section 6.3.1')

    args, _ = parser.parse_known_args(args)
    return args


def get_selection_args(args=None):
    '''
    Arguments for sample selection
    '''
    parser = ArgumentParser()

    parser.add_argument('--selection', type=str, default='hvi', 
        help='type of selection method for new batch')
    parser.add_argument('--batch-size', type=int, default=10, 
        help='size of the selected batch in one iteration')

    args, _ = parser.parse_known_args(args)
    return args


def get_args():
    '''
    Get arguments from all components
    You can specify args-path argument to directly load arguments from specified yaml file
    '''
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None,
        help='used for directly loading arguments from path of argument file')
    args, _ = parser.parse_known_args()

    if args.args_path is None:

        general_args = get_general_args()
        surroagte_args = get_surroagte_args()
        acquisition_args = get_acquisition_args()
        solver_args = get_solver_args()
        selection_args = get_selection_args()

        framework_args = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
        }

    else:
        
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)
        
        general_args = Namespace(**all_args['general'])
        framework_args = all_args.copy()
        framework_args.pop('general')

    return general_args, framework_args
