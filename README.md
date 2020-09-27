# Diversity-Guided Efficient Multi-Objective Optimization (DGEMO)

Algorithm framework for multi-objective Bayesian optimization, including the official implementation of DGEMO and re-implementations of other popular MOBO algorithms.

## Key Features

- **Algorithm**: Support DGEMO, TSEMO, USeMO-EI, MOEA/D-EGO, ParEGO, NSGA-II and custom algorithms. See *mobo/algorithms.py* to select proper algorithm to use / define your own algorithm.
  - **Test problem**: Support: ZDT1-3, DTLZ1-6, OKA1-2, VLMOP2-3, RE. Also constraint handling is implemented, if the problem constraints are properly defined according to [Pymoo Problem Definition](https://pymoo.org/problems/custom.html) (see the "G" functions).
- **Surrogate model**: Support Gaussian process as surrogate model to evaluate samples, or sampled functions by Thompson Sampling from the fitted Gaussian process. See *mobo/surrogate_model/*.
- **Acquisition function**: Support PI, EI, UCB and identity function as acquisition, see *mobo/acquisition.py*.
- **Solver**: Support using NSGA-II, MOEA/D and ParetoDiscovery [[Schulz et al. 2018]](https://dl.acm.org/doi/10.1145/3197517.3201385) to solve the multi-objective surrogate problem. See *mobo/solver/*.
- **Selection**: Support: HVI, uncertainty, random, etc. as criterion for selecting next (batch of) samples to evaluate on the real problem. See *mobo/selection.py*.

For DGEMO, we use Gaussian process as surrogate model, identity function as acquisition function, ParetoDiscovery as multi-objective solver and our diversity-guided selection algorithm.

## Code Structure

```
baselines/ --- MOO baseline algorithms: NSGA-II
mobo/
 ├── solver/ --- multi-objective solvers
 ├── surrogate_model/ --- surrogate models
 ├── acquisition.py --- acquisition functions
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── mobo.py --- main pipeline of multi-objective bayesian optimziation
 ├── selection.py --- selection methods for new samples
 ├── surrogate_problem.py --- multi-objective surrogate problem
 ├── transformation.py --- normalizations on data
 └── utils.py --- utility functions
problems/ --- multi-objective problem definitions
scripts/ --- scripts for batch experiments
visualization/ --- performance visualization
main.py --- main execution file for MOBO algorithms
```

## Requirements

- Python version: tested in Python 3.7.7

- Operating system: tested in Ubuntu 18.04

- Install the environment by [conda](https://www.anaconda.com/) and activate:

  ```
  conda env create -f environment.yml
  conda activate mobo
  ```

  Install other dependencies by pip:

  ```
  pip install -r requirements.txt
  ```

- If the pip installed pymoo is not compiled (will raise warning when running the code), you can clone the pymoo github repository, compile and install this module as described [here](https://pymoo.org/installation.html#development), to gain extra speed-up compared to uncompiled version.

## Getting Started

### Basic usage

Run the main file with python with specified arguments:

```
python main.py --problem dtlz1 --n-var 6 --n-obj 2 --n-iter 20
```

If you don't understand the meaning of the arguments, see *argument.py* or:

```
python main.py --help
```

### Parallel experiment

Run the script file with python, for example:

```
python scripts/run.py --problem dtlz1 zdt1 --algo dgemo parego --n-seed 10 --n-process 8
```

This will run algorithm DGEMO and ParEGO on DTLZ1 and ZDT1 problems, with 10 different random seeds. So in total there'll be 2 * 2 * 10 = 40 different experiments. Since we specified 8 parallel processes to use, these 40 experiments will then be distributed to 8 processes to run in order. Whenever a process finishes its current experiment will be filled by another experiment that hasn't been run.

## Performance

![performance](performance.png)

To reproduce this result, run the following command:

```
python scripts/run.py --algo nsga2 parego moead-ego tsemo usemo-ei dgemo --problem zdt1 zdt2 zdt3 dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 oka1 oka2 vlmop2 vlmop3 re1 re2 re3 re4 re5 re6 re7 --batch-size 10 --n-iter 20 --n-seed 10
```

This command produces the results on all problems using all algorithms with 10 different random seeds. In total there are 20 * 6 * 10 = 1200 individual experiments, with 20 iterations in each experiment, which could probably take hours or days to finish, depending on the hardware configurations and parallelization level. See Appendix C.2 in our paper for runtime (speed) statistics of different algorithms and our hardware platform for producing that statistics.

To visualize this figure, run the following command:

```
python visualization/visualize_hv_all.py
```

## Result

The optimization results are saved in csv format and the arguments are saved as a yaml file. They are stored under the folder:

```
result/{problem}/{subfolder}/{algo}-{exp-name}/{seed}/
```

By default the *subfolder* name is 'default'. If no experiment name (*exp-name*, which is used to distinguish between experiments using same problem and same algorithm but has other different detailed parameters) specified, then the folder would be:

```
result/{problem}/{subfolder}/{algo}/{seed}/
```

Then, under that folder, the name of csv files would be:

```
(EvaluatedSamples/ParetoFrontEvaluated/ParetoFrontApproximation).csv
```

The name of the argument yaml file is `args.yml`.

*Explanation --- problem: problem name, algo: algorithm name, exp-name: experiment name, seed: random seed used*

## Visualization

For batch visualization of hypervolume curve produced from multiple experiments, run for example:

```
python visualization/visualize_hv_batch.py --problem zdt1 --algo dgemo --n-seed 10
```

This command will produce a figure of a hypervolume curve from experiments ran on ZDT1 problem, using DGEMO as algorithm, averaged across 10 random seeds. 

The batch visualization is also compatible with multiple algorithms for plotting multiple hypervolume curves on a single figure for comparison. Also it can produce multiple figures for multiple problems at the same time, for example:

```
python visualization/visualize_hv_batch.py --problem zdt1 zdt2 zdt3 --algo dgemo tsemo parego --n-seed 10
```

This command will produce 3 figures for ZDT1, ZDT2 and ZDT3 problems respectively. In each figure, there are three hypervolume curves from experiments using DGEMO, TSEMO and ParEGO respectively, averaged across 10 random seeds.

Note if you don't specify `--problem` or `--algo` arguments, it will automatically find all the problems or algorithms you have in the result folder.

## Citation

To be added.