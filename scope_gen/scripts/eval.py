import os
import pickle
import psutil

from scope_gen.algorithms.base import run_experiment

def eval_all(cfgs,
             n_iterations = 10,
             ratio_cal_test = 1.0,
             verbose=True):
    if verbose:
        print(f"Running evaluation for {len(cfgs)} SCOPE-Gen configurations.")

    for cfg in cfgs:
        eval(n_iterations = 10, ratio_cal_test = 1.0, **cfg)
        if verbose:
            print(f"Finished evaluation for SCOPE-Gen configuration {cfg}.")

def eval(name,
         score,
         stages,
         data_dir,
         adm,
         adm_threshold,
         mass_bins,
         alpha_grid,
         n_iterations = 10,
         ratio_cal_test = 1.0, 
         return_std_coverages=True,
         alpha_params=None,
         debug=False,
         verbose=True
         ):
    # run experiments
    K = len(stages)

    if verbose:
        print(f"Running {len(alpha_grid)} experiments.")
    for alpha in alpha_grid:
        if verbose:
            print(f"Running experiment with alpha {alpha}.")
        
        experiment_config = {
            "data_dir": data_dir,
            "ratio_cal_test": ratio_cal_test,
            "n_iterations": n_iterations,
            "split_ratios": [1/K] * K,
            "alpha": alpha,
            "score": score,
            "verbose": verbose,
            "debug": debug,
            "stages": stages,
            "name": name,
            "adm": adm,
            "adm_threshold": adm_threshold,
            "mass_bins": mass_bins,
            "return_std_coverages": return_std_coverages,
            "alpha_params": alpha_params
        }
        
        run_experiment(**experiment_config)
