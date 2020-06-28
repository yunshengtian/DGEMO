import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from argparse import ArgumentParser
from problems.common import build_problem


def get_ref_point(problem, n_var=6, n_obj=2, n_init_sample=50, seed=0):

    np.random.seed(seed)
    _, _, _, Y_init = build_problem(problem, n_var, n_obj, n_init_sample)

    return np.max(Y_init, axis=0)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n-var', type=int, default=6)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-init-sample', type=int, default=50)
    args = parser.parse_args()

    ref_point = get_ref_point(args.problem, args.n_var, args.n_obj, args.n_init_sample)

    print(f'Reference point: {ref_point}')