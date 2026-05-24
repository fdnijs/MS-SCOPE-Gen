import os

import sys
sys.path.append(os.path.abspath("../../.."))
from scope_gen.scripts.format_data import format_data
from scope_gen.de_novo_molecules.paths import DATA_DIR



if __name__ == "__main__":
    input_dir = os.path.join(DATA_DIR, "finetune-original/beam100")
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.4)
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.675)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=10)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=5)
    
    input_dir = os.path.join(DATA_DIR, "finetune-mces1/beam100")
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.4)
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.675)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=10)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=5)
    
    input_dir = os.path.join(DATA_DIR, "finetune-random/beam100")
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.4)
    format_data(data_dir=input_dir, adm="tani", adm_threshold=0.675)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=10)
    format_data(data_dir=input_dir, adm="mces", adm_threshold=5)
    