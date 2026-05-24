"""Script that evaluates all baselines at once."""

import os
import sys
sys.path.append(os.path.abspath("../../.."))
from scope_gen.de_novo_molecules.paths import CONFIG_DIR, DATA_DIR
from scope_gen.utils import load_config_from_json, load_configs_from_jsonl, set_seed
from scope_gen.scripts.eval import eval_all

VERBOSE = True

if __name__ == "__main__":
    set_seed(0)
    # load json lines configs for both methods
    cfgs_eval = load_configs_from_jsonl(CONFIG_DIR + "/eval.jsonl")
    eval_all(cfgs_eval,
             n_iterations = 10,
             ratio_cal_test = 1.0,
             verbose=VERBOSE
            )