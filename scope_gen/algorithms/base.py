"""This script contains the main functions for the SCOPE-Gen algorithm."""

import time
import multiprocessing
import numpy as np
import pandas as pd
import psutil
import sys
import os 
import pickle

from scope_gen.data.base import split_data_idxs
from scope_gen.calibrate.base import get_percentile
from scope_gen.order_funcs import quality_order_func, distance_order_func
from scope_gen.models.pipelines import (MinimalPredictionPipeline, 
                                        GenerationPredictionPipeline, 
                                        FilterPredictionPipeline, 
                                        RemoveDuplicatesPredictionPipeline)
from scope_gen.calibrate.score_computation import ScoreComputer
from scope_gen.nc_scores import (MaxScore, 
                                 MinScore, 
                                 DistanceScore, 
                                 CountScore, 
                                 SumScore)
from scope_gen.utils import store_results


SCORES_TABLE = {
                    "count": CountScore, 
                    "max": MaxScore,
                    "sum": SumScore
                }

SCORE_FUNC_QUALITY = MinScore()
SCORE_FUNC_DIST = DistanceScore()
ORDER_FUNC_QUALITY = quality_order_func
ORDER_FUNC_DIST = distance_order_func

# actually not required, but for the sake of consistency
MULTIPROCESSING = False


def create_scope_gen_pipeline(data, 
                              split_ratios, 
                              alphas, 
                              score, 
                              data_splitting=True, 
                              stages=None, 
                              count_adm=True):
    """Create a SCOPE-Gen pipeline."""
    # Split the dataset into calibration for generation, calibration for quality pruning, and data for diversity pruning
    if stages is None:
        from scope_gen.data.meta_data import STAGES
        stages = STAGES
    if data_splitting:
        data_split_idxs = split_data_idxs(len(data), split_ratios)
    first_adms = []
    conformal_ps = []
    adjust_for_dupl = False
    if "remove_dupl" in stages:
        adjust_for_dupl = True
    score_func = SCORES_TABLE[score]()
    pipeline = MinimalPredictionPipeline(data)
    for i, stage in enumerate(stages):
        if not stage == "remove_dupl":
            conformal_p, first_adm = _calibrate_pipeline(pipeline, stage, stages, score_func, 
                                                         adjust_for_dupl, count_adm, data_split_idxs, alphas)
            if count_adm:
                first_adms.append(first_adm)
            conformal_ps.append(conformal_p)
        # Update the prediction pipeline
        pipeline = _update_pipeline(pipeline, stage, score_func, conformal_p)
    out = {"pipeline": pipeline, "conformal_ps": conformal_ps}

    if count_adm:
        out["first_adms"] = np.stack(first_adms, axis=0).flatten()
    return out


def _calibrate_pipeline(pipeline, stage, stages, score_func, adjust_for_dupl, count_adm, data_split_idxs, alphas):
    score_computer = _initialize_score_computer(stage, pipeline, score_func, adjust_for_dupl)
    first_adm = None
    if count_adm:
        cal_scores, first_adm = score_computer.compute_scores(idxs=data_split_idxs[stages.index(stage)], return_idxs=True)
    else:
        cal_scores = score_computer.compute_scores(idxs=data_split_idxs[stages.index(stage)])
    # Compute the conformal quantile for generation
    conformal_p = get_percentile(cal_scores, alphas[stages.index(stage)])
    return conformal_p, first_adm


def _update_pipeline(pipeline, stage, score_func, conformal_p):
    if stage == "generation":
            pipeline = GenerationPredictionPipeline(pipeline, conformal_p, score_func)
    else:
        if stage == "quality":
            pipeline = FilterPredictionPipeline(pipeline, conformal_p, SCORE_FUNC_QUALITY, ORDER_FUNC_QUALITY)
        elif stage == "diversity":
            pipeline = FilterPredictionPipeline(pipeline, conformal_p, SCORE_FUNC_DIST, ORDER_FUNC_DIST)
        elif stage == "remove_dupl":
            pipeline = RemoveDuplicatesPredictionPipeline(pipeline)
    return pipeline


def _initialize_score_computer(stage, pipeline, score_func, adjust_for_dupl):
    if stage == "generation":
        score_computer = ScoreComputer(pipeline, score_func=score_func, order_func=None, 
                                        adjust_for_dupl=adjust_for_dupl)
    elif stage == "quality":
        score_computer = ScoreComputer(pipeline, score_func=SCORE_FUNC_QUALITY, 
                                        order_func=ORDER_FUNC_QUALITY, adjust_for_dupl=adjust_for_dupl)
    elif stage == "diversity":
        score_computer = ScoreComputer(pipeline, score_func=SCORE_FUNC_DIST, 
                                        order_func=ORDER_FUNC_DIST, adjust_for_dupl=adjust_for_dupl)
    else:
        raise ValueError(f"Stage {stage} not recognized.")

    return score_computer


def test(data, pipeline, return_std_coverages=False):
    """Test the pipeline."""
    pipeline.data = data
    coverages = []
    sizes = []
    set_positions = []
    for i in range(len(data)):
        prediction_set = pipeline.generate(i)
        if prediction_set is None:
            # "return everything" (reject)
            coverages.append(1)
            sizes.append(np.inf)
            set_positions.append(None)
        else:
            coverages.append(int(any(prediction_set["labels"] == 1)))
            sizes.append(len(prediction_set["labels"]))
            if "set_pos" in prediction_set:
                set_positions.append(prediction_set["set_pos"].copy())
            else:
                set_positions.append(None) 
    if return_std_coverages:
        return coverages, sizes, set_positions, coverages
    return coverages, sizes, set_positions


def experiment_iteration(args):
    (bins, masses, ids, data, ratio_cal_test, split_ratios, alphas,
     score, stages, idx, return_std_coverages, label_map) = args    
    COUNT_ADMS = True
    def get_masses(idxs):
        return [masses[i] for i in idxs]

    results = {
        "coverages": [],
        "sizes": [],
        "first_adms": [],
        "std_coverages": [],
        "mass_bins": [],
        "gt_masses": [],
        "set_positions_data_idx": [] 
    }

    cal_fraction = ratio_cal_test / (1 + ratio_cal_test)

    for b_idx, bin_indices in enumerate(bins):
        if len(bin_indices) < 2:
            print(f"Error: mass bin {b_idx} has <2 samples. Cannot split.")
            sys.exit(1)

        bin_masses = masses[bin_indices]
        mass_min = float(bin_masses.min())
        mass_max = float(bin_masses.max())

        rng = np.random.default_rng(seed=(idx, b_idx))
        perm = rng.permutation(len(bin_indices))

        split = int(cal_fraction * len(perm))
        split = max(1, min(split, len(perm) - 1))

        cal_idxs = [bin_indices[i] for i in perm[:split]]
        test_idxs = [bin_indices[i] for i in perm[split:]]

        if label_map is not None:
            seen_idxs = [i for i in test_idxs if label_map.get(ids[i], "unseen") == "seen"]
            unseen_idxs = [i for i in test_idxs if label_map.get(ids[i], "unseen") == "unseen"]
        else:
            seen_idxs = test_idxs
            unseen_idxs = []
        seen_masses = get_masses(seen_idxs)
        unseen_masses = get_masses(unseen_idxs)

        data_cal = [data[i] for i in cal_idxs]
        out = create_scope_gen_pipeline(data=data_cal, 
                                        split_ratios=split_ratios, 
                                        alphas=alphas,
                                        score=score, 
                                        data_splitting=True, 
                                        stages=stages, 
                                        count_adm=COUNT_ADMS)
        def run_test(idxs):
            if len(idxs) == 0:
                empty_result = (np.nan, 0, []) if not return_std_coverages else (np.nan, 0, [], np.nan)
                return empty_result

            if return_std_coverages:
                coverages, sizes, std_coverages, set_positions = test([data[i] for i in idxs],
                                                                    out["pipeline"],
                                                                    return_std_coverages=True)
                return coverages, sizes, std_coverages, set_positions
            else:
                coverages, sizes, set_positions = test([data[i] for i in idxs],
                                                    out["pipeline"])
                return coverages, sizes, set_positions
            
        seen_res = run_test(seen_idxs)
        unseen_res = run_test(unseen_idxs)

        results["first_adms"].append(out["first_adms"])
        results["coverages"].append((seen_res[0], unseen_res[0]))
        results["sizes"].append((seen_res[1], unseen_res[1]))
        results["set_positions_data_idx"].append(
            (dict(zip(seen_idxs, seen_res[2])),
            dict(zip(unseen_idxs, unseen_res[2])))
        )
        results["mass_bins"].append({
            "bin_id": b_idx,
            "min_mass": mass_min,
            "max_mass": mass_max,
            "n_samples": len(bin_indices),
            "n_cal": len(cal_idxs),
            "n_test": len(test_idxs),
        })
        results["gt_masses"].append((seen_masses, unseen_masses))
        if return_std_coverages:
            results["std_coverages"].append(
                (seen_res[3], unseen_res[3])
            )
    return results


def run_experiment(data_dir,
                   ratio_cal_test,
                   n_iterations,
                   split_ratios, 
                   alpha, 
                   score, 
                   stages, 
                   name, 
                   adm,
                   adm_threshold,
                   mass_bins,
                   verbose=False,
                   debug=False,
                   alpha_params=None, 
                   return_std_coverages=False):
    
    """Fits the SCOPE-Gen pipeline to the data for many 
    iterations and one evaluation per fit."""
    
    K = len(split_ratios)

    if alpha_params is None:
        alpha_params = {"M" : 5, "parts" : [5 - (K - 1)] + [1 for _ in range(K-1)]}
    
    assert len(alpha_params["parts"]) == K
    
    alphas = compute_alphas(alpha, alpha_params)
    
    labels = pd.read_csv(f"{data_dir}/labels.tsv", sep="\t")

    has_label = "label" in labels.columns
    if has_label:
        label_map = dict(zip(labels["id"], labels["label"]))

    ids = labels["id"].values
    masses = labels["parent_mass"].values

    sorted_idx = np.argsort(masses)

    bins = [
        sorted_idx[chunk]
        for chunk in np.array_split(np.arange(len(sorted_idx)), mass_bins)
    ]

    data_path = os.path.join(data_dir, f"{adm}_{adm_threshold}.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    assert len(data) == len(masses) == len(ids)

    args = [
        (bins, masses, ids, data, ratio_cal_test, split_ratios, alphas, score,
        stages, i, return_std_coverages, label_map if has_label else None)
        for i in range(n_iterations)
    ]
    
    num_processes = 8
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(experiment_iteration, args)

    # Aggregate results from all iterations
    aggregate_results = {
        "coverages": [],
        "sizes": [],
        "first_adms": [],
        "std_coverages": [],
        "mass_bins": [],
        "gt_masses": [],
        "set_positions_data_idx": []
    }

    for result in results:
        aggregate_results["coverages"].extend(result["coverages"])
        aggregate_results["sizes"].extend(result["sizes"])
        aggregate_results["first_adms"].extend(result["first_adms"])
        aggregate_results["mass_bins"].extend(result["mass_bins"])
        aggregate_results["gt_masses"].extend(result["gt_masses"])
        aggregate_results["set_positions_data_idx"].extend(result["set_positions_data_idx"])
        if return_std_coverages:
            aggregate_results["std_coverages"].extend(result["std_coverages"])

    # Store results to disk unless debugging
    type = None
    if not debug:
        store_results(
            data_dir, name, alpha, score, mass_bins,
            aggregate_results["coverages"],
            aggregate_results["sizes"],
            aggregate_results["first_adms"],
            aggregate_results["mass_bins"],
            aggregate_results["gt_masses"],
            ratio_cal_test,
            type=type,
            std_coverages=aggregate_results["std_coverages"],
            set_positions_data_idx=aggregate_results["set_positions_data_idx"]  # NEW!
        )


def compute_alphas(alpha, alpha_params):
    if len(alpha_params["parts"]) > 1:
        assert alpha_params["M"] == sum(alpha_params["parts"])
        chunk = (1 - alpha)**(1/alpha_params["M"])
        alphas = [(1 - chunk**alpha_params["parts"][j]) for j in range(len(alpha_params["parts"]))]
    else:
        alphas = [alpha]
    return alphas