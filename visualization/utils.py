import os
import pandas as pd


def get_problem_dir(args):
    '''
    Get problem directory under result directory
    '''
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    problem_dir = os.path.join(result_dir, args.problem, args.subfolder)
    return problem_dir


def get_problem_names():
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    return sorted(os.listdir(top_dir))


def get_algo_names(args):
    '''
    Get algorithm name / names for comparison, also check if specified algorithm is valid
    '''
    problem_dir = get_problem_dir(args)

    algo_names = set()
    for algo_name in os.listdir(problem_dir):
        if algo_name != 'TrueParetoFront.csv':
            algo_names.add(algo_name)
    if len(algo_names) == 0:
        raise Exception(f'cannot found valid result file under {problem_dir}')

    # if algo argument not specified, return all algorithm names found under the problem directory
    if args.algo is None:
        args.algo = list(algo_names)

    algo_names = args.algo
    return sorted(algo_names)


defaultColors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#fd3216',  # Light24 list
    '#ea1b85',  # more below
    '#7d7803',
    '#ff8fa6',
    '#aeeeee',
    '#7a6ba1',
    '#820028',
    '#d16c6a',
]
