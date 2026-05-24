import argparse
import os
import pandas as pd
from pathlib import Path
import copy
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem

import sys
sys.path.append("./mist/src")
from mist.utils.plot_utils import *
import mist.models.base as base
import mist.data.datasets as datasets
import mist.data.featurizers as featurizers

class MISTPredictor:
    def __init__(self, fp_ckpt, res_dir, mgf_input, labels):
        self.fp_ckpt = fp_ckpt
        self.res_dir = Path(res_dir)
        self.mgf_input = mgf_input
        self.labels = labels
        self.res_dir.mkdir(exist_ok=True)
        self.subform_dir = self.res_dir / "subformulae/default_subformulae"

        self.device = torch.device("cuda:0")
        self.test_dataset = None

        self.load_model()

    def load_model(self):
        fp_model = torch.load(self.fp_ckpt, map_location=self.device)
        main_hparams = fp_model["hyper_parameters"]
        self.kwargs = copy.deepcopy(main_hparams)
        self.kwargs['device'] = "cuda:0"
        self.kwargs['num_workers'] = 0
        self.kwargs['subform_folder'] = self.subform_dir
        self.kwargs['labels_file'] = self.labels

        self.model = base.build_model(**self.kwargs)
        self.model.load_state_dict(fp_model["state_dict"])
        self.model = self.model.to(self.device)
        self.model = self.model.eval()


    def prepare_dataset(self):
        self.kwargs["spec_features"] = self.model.spec_features(mode="test") # 'peakformula_test'
        self.kwargs['mol_features'] = "none"
        self.kwargs['allow_none_smiles'] = True
        paired_featurizer = featurizers.get_paired_featurizer(**self.kwargs)

        spectra_mol_pairs = datasets.get_paired_spectra(**self.kwargs)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        self.test_dataset = datasets.SpectraMolDataset(
            spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **self.kwargs
        )

    def predict(self):
        self.prepare_dataset()
        output_preds = (
            self.model.encode_all_spectras(self.test_dataset, no_grad=True, **self.kwargs).cpu().numpy()
        )
        output_names = self.test_dataset.get_spectra_names()
        return output_preds, output_names

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mist-ckpt", default="mist_fp.ckpt", help="Checkpoint filename")
    p.add_argument("--dataset", default="MassSpecGym")
    args = p.parse_args()
    
    dataset_name = args.dataset
    ckpt_name = args.mist_ckpt
    ckpt_base = os.path.splitext(ckpt_name)[0]

    fp_ckpt = f"data/{dataset_name}/mist/ckpts/{ckpt_name}"
    res_dir = f"data/{dataset_name}/mist/"
    mgf_input = f"data/{dataset_name}/{dataset_name}.mgf"
    labels = f"data/{dataset_name}/{dataset_name}_labels.tsv"

    predictor = MISTPredictor(fp_ckpt, res_dir, mgf_input, labels)

    output_preds, output_names = predictor.predict()
    print(output_preds.shape, len(output_names))

    for threshold in [0.11]:
        indices_list = [np.where(row > threshold)[0].tolist() for row in output_preds]
        name_fps_keys = {name: fps for name, fps in zip(output_names, indices_list)}

        if dataset_name == "MassSpecGym":
            df = pd.read_csv("data/MassSpecGym/MassSpecGym.tsv", sep='\t')
        else:
            raise ValueError("Unknown dataset name")
        print(f"Before processing, the number of entries is: {len(df)}")
        # Add FPS column/SMILES column
        df['fps'] = ''
        df['SMILES'] = ''
        
        # Record the indices of rows that need to be deleted (those where FPS or SMILES generation failed)
        to_drop = []
        for idx, row in df.iterrows():
            identifier_key = "identifier" if dataset_name == "MassSpecGym" else "name"
            identifier = row[identifier_key]
            smiles = row['smiles']
            if identifier not in name_fps_keys:
                to_drop.append(idx)
                print(f"identifier {identifier} not in name_fps_keys")
                continue
            fps = name_fps_keys[identifier]
            df.at[idx, 'fps'] = "".join([f"<fp{fp:04d}>" for fp in fps])
            try:
                # Convert to standard SMILES
                mol = Chem.MolFromSmiles(smiles)
                Chem.RemoveStereochemistry(mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                df.at[idx, 'SMILES'] = canonical_smiles
            except:
                print(f"Error in SMILES: {smiles}")
                to_drop.append(idx)
                continue
        # Delete rows where SELFIES generation failed
        df = df.drop(index=to_drop)
        print(f"Final number of entries: {len(df)}")

        out_path = f"./data/{dataset_name}/{dataset_name}_{ckpt_base}_threshold_{threshold}.tsv"
        df.to_csv(out_path, sep='\t', index=False)
        print(f"Saved: {out_path}")