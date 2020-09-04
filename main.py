import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up
import numpy as np
from problems.common import build_problem
from mobo.algorithms import get_algorithm
from visualization.data_export import DataExport
from arguments import get_args
from utils import save_args, setup_logger

'''
Main entry for MOBO execution
'''

def main():
    # load arguments
    args, framework_args = get_args()

    # set seed
    np.random.seed(args.seed)

    # build problem, get initial samples
    problem, true_pfront, X_init, Y_init = build_problem(args.problem, args.n_var, args.n_obj, args.n_init_sample, args.n_process)
    args.n_var, args.n_obj = problem.n_var, problem.n_obj

    # initialize optimizer
    optimizer = get_algorithm(args.algo)(problem, args.n_iter, args.ref_point, framework_args)

    # save arguments & setup logger
    save_args(args, framework_args)
    logger = setup_logger(args)
    print(problem, optimizer, sep='\n')
    
    # initialize data exporter
    exporter = DataExport(optimizer, X_init, Y_init, args)

    # optimization
    solution = optimizer.solve(X_init, Y_init)

    # export true Pareto front to csv
    if true_pfront is not None:
        exporter.write_truefront_csv(true_pfront)

    for _ in range(args.n_iter):
        # get new design samples and corresponding performance
        X_next, Y_next = next(solution)
        
        # update & export current status to csv
        exporter.update(X_next, Y_next)
        exporter.write_csvs()

    # close logger
    if logger is not None:
        logger.close()


if __name__ == '__main__':
    main()