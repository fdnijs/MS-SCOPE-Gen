import typing as T
from rdkit import Chem
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from tqdm import tqdm

import os
import pickle

import sys
sys.path.append('.')
from src.eval.utils import morgan_fp, MyopicMCES

RDLogger.DisableLog('rdApp.*')

class MoleculeEvaluator:
    def __init__(self, mces: bool = False):
        self.mces = mces
        if mces:
            print("MCES is enabled")
            self.myopic_mces = MyopicMCES()
            self.mces_cache = {}
        self.fps_cache = {}

    def load_mces_cache(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.mces_cache = pickle.load(f)

    def save_mces_cache(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.mces_cache, f)

    def calculate_mces(self, pred_true_smiles: T.Tuple[str, str]) -> float:
        pred_smi, true_smi = pred_true_smiles
        if pred_smi is None or true_smi is None:
            return 100
        try:
            key = tuple(sorted([pred_smi, true_smi]))
            if key not in self.mces_cache:
                mce_val = self.myopic_mces(true_smi, pred_smi)
            else: 
                mce_val = self.mces_cache[key]
        except Exception as e:
            print(f'ERROR: {true_smi} {pred_smi} MCES {e}')
            mce_val = 100
        return mce_val, key

    def calculate_tanimoto(self, pred_true_mols_smiles: T.Tuple[Chem.Mol, Chem.Mol, str, str]) -> float:
        pred_mol, true_mol, pred_smi, true_smi = pred_true_mols_smiles
        if pred_smi is None or true_smi is None:
            return 0.0
        try:
            if pred_smi not in self.fps_cache:
                self.fps_cache[pred_smi] = morgan_fp(pred_mol, to_np=False)
            if true_smi not in self.fps_cache:
                self.fps_cache[true_smi] = morgan_fp(true_mol, to_np=False)
            sim = TanimotoSimilarity(self.fps_cache[true_smi], self.fps_cache[pred_smi])
        except Exception as e:
            print(f"Tanimoto error: {e}")
            sim = 0.0
        return sim

    def is_smiles_valid_with_rdkit(self, smiles_str: str) -> T.Tuple[T.Optional[Chem.Mol], T.Optional[str]]:
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return None, None
            return mol, Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            return None, None

    def calculate_mol_from_smiles(self, pred_true_smiles: T.Tuple[str, str]) -> T.Optional[T.List[T.List]]:
        pred_smiles, true_smiles = pred_true_smiles
        pred_mol, canonical_pred_smiles = self.is_smiles_valid_with_rdkit(pred_smiles)
        true_mol, canonical_true_smiles = self.is_smiles_valid_with_rdkit(true_smiles)
        if pred_mol and true_mol:
            return [[pred_mol, canonical_pred_smiles], [true_mol, canonical_true_smiles]]
        return [[None, None], [None, None]]

    def evaluate_de_novo_step_smiles(self, smiles_pred: T.List[str], smiles_true: T.List[str], return_all = False, cpu_count = 8) -> T.Dict[str, float]:
        print("Multi-processing to calculate mol and canonical smiles: ", cpu_count)

        valid_mols_pred, valid_mols_true, valid_smiles_pred, valid_smiles_true = [], [], [], []

        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_mol_from_smiles, zip(smiles_pred, smiles_true))
            mols_results = mols_future.result()
            with tqdm(total=len(smiles_true), desc="Calculate mol and canonical smiles: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        valid_mols_pred.append(result[0][0])
                        valid_mols_true.append(result[1][0])
                        valid_smiles_pred.append(result[0][1])
                        valid_smiles_true.append(result[1][1])
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error, flush=True)
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)

        if  len(valid_mols_pred) == 0:
            return {
                "top1_valid_mols": 0,
                "top1_mces_dist": 100,
                "top1_tanimoto_sim": 0,
                "top1_mol_accuracy": 0,
            }

        metric_vals = {
            "top1_valid_mols": len(valid_mols_pred) / len(smiles_pred),
            "top1_mces_dist": 100,
            "top1_tanimoto_sim": 0,
            "top1_mol_accuracy": 0,
        }

        tanimoto_sims = []
        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_tanimoto, zip(valid_mols_pred, valid_mols_true, valid_smiles_pred, valid_smiles_true))
            mols_results = mols_future.result()
            with tqdm(total=len(valid_smiles_true), desc="Calculate valid tanimoto: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        tanimoto_sims.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error, flush=True)
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        metric_vals["top1_tanimoto_sim"] =float(np.mean(tanimoto_sims))
        
        mces_dists = []
        if self.mces:
            with ProcessPool(max_workers=cpu_count) as pool:
                mces_future = pool.map(self.calculate_mces, zip(valid_smiles_pred, valid_smiles_true))
                mces_results = mces_future.result()
                with tqdm(total=len(valid_smiles_true), desc="Calculate MCES: ") as progress_bar:
                    while True:
                        try:
                            mce_val, key = next(mces_results)
                            mces_dists.append(mce_val)
                            if key not in self.mces_cache:
                                self.mces_cache[key] = mce_val
                        except StopIteration:
                            break
                        except Exception as error:
                            print(error, flush=True)
                            exit()
                        progress_bar.update(1)
            metric_vals["top1_mces_dist"] = float(np.mean(mces_dists))

        correct_cnt = sum(1 for pred_smi, true_smi in zip(valid_smiles_pred, valid_smiles_true) if pred_smi == true_smi)
        metric_vals["top1_mol_accuracy"] = correct_cnt / len(valid_smiles_pred)

        for key in metric_vals:
            metric_vals[key] = round(metric_vals[key], 4)

        if return_all:
            return metric_vals, tanimoto_sims, mces_dists
        else:
            return metric_vals
            
    

class TopkMoleculeEvaluator:
    def __init__(self, mces: bool = False):
        self.fps_cache = {}
        self.mces = mces
        if mces:
            print("MCES is enabled")
            self.myopic_mces = MyopicMCES()
            self.mces_cache = {}

    def load_mces_cache(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.mces_cache = pickle.load(f)

    def save_mces_cache(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.mces_cache, f)

    def get_canonical_smiles_from_smiles(self, pred_true_smiles):
        def get_canonical_smiles(smiles_str):
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                return None
        pred_smiles_topk, true_smiles = pred_true_smiles
        pred_smiles_topk = [get_canonical_smiles(smiles) for smiles in pred_smiles_topk]
        true_smiles = get_canonical_smiles(true_smiles)
        if true_smiles is None: return None
        return pred_smiles_topk, true_smiles
    
    def calculate_tanimoto(self, pred_true_smiles):
        pred_smi_topk, true_smi = pred_true_smiles
        k = len(pred_smi_topk)
        if true_smi is None:
            return k * [0.0]
        topk_tanimoto_sims = []
        for pred_smi in pred_smi_topk:
            if pred_smi is None:
                topk_tanimoto_sims.append(0.0)
                continue
            try:
                if pred_smi not in self.fps_cache:
                    pred_mol = Chem.MolFromSmiles(pred_smi)
                    self.fps_cache[pred_smi] = morgan_fp(pred_mol, to_np=False)
                if true_smi not in self.fps_cache:
                    true_mol = Chem.MolFromSmiles(true_smi)
                    self.fps_cache[true_smi] = morgan_fp(true_mol, to_np=False)
                sim = TanimotoSimilarity(self.fps_cache[true_smi], self.fps_cache[pred_smi])
                topk_tanimoto_sims.append(sim)
            except Exception as e:
                print(f"Tanimoto error: {e}")
        if len(topk_tanimoto_sims) == 0:
            return k * [0.0]
        return topk_tanimoto_sims
    
    def calculate_mces(self, pred_true_smiles):
        pred_smi_topk, true_smi = pred_true_smiles
        k = len(pred_smi_topk)
        if true_smi is None:
            return k * [100.0]
        topk_mces = []
        keys = []
        for pred_smi in pred_smi_topk:
            if pred_smi is None:
                topk_mces.append(100.0)
                continue
            try:
                key = tuple(sorted([pred_smi, true_smi]))
                if key not in self.mces_cache:
                    mce_val = self.myopic_mces(true_smi, pred_smi)
                else:
                    mce_val = self.mces_cache[key]
                keys.append(key)
                topk_mces.append(mce_val)
            except Exception as e:
                print(f'MCES error: {e}')
        if len(topk_mces) == 0:
            return k * [100.0]
        return topk_mces, keys
    
    def evaluate_de_novo_step_smiles_top_k(self, smiles_pred_topk: T.List[T.List[str]], smiles_true: T.List[str], return_all = False, save_path: str = None) -> T.Dict[str, float]:
        cpu_count = 20
        print("Multi-processing to calculate canonical smiles: ", cpu_count)

        canonical_smiles_pred_topk, canonical_smiles_true = [], []
        with ProcessPool(max_workers=cpu_count) as pool:
            smiles_future = pool.map(self.get_canonical_smiles_from_smiles, zip(smiles_pred_topk, smiles_true))
            smiles_results = smiles_future.result()
            with tqdm(total=len(smiles_true), desc="Calculate canonical smiles: ") as progress_bar:
                while True:
                    try:
                        result = next(smiles_results)
                        if result:
                            canonical_smiles_pred_topk.append(result[0])
                            canonical_smiles_true.append(result[1])
                    except StopIteration:
                        break
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        valid_cnt, total_cnt = 0, 0
        for item in canonical_smiles_pred_topk:
            for smiles in item:
                total_cnt += 1
                if smiles is not None:
                    valid_cnt += 1
        if valid_cnt == 0:
            return {
                "topk_valid_mols": 0,
                "topk_mces_dist": 100,
                "topk_tanimoto_sim": 0,
                "topk_mol_accuracy": 0,
            }
        metric_vals = {
            "topk_valid_mols": valid_cnt / total_cnt if total_cnt > 0 else 0,
            "topk_mces_dist": 100,
            "topk_tanimoto_sim": 0,
            "topk_mol_accuracy": 0,
        }
        
        tanimoto_sims = []
        all_tanimoto_sims = []
        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_tanimoto, zip(canonical_smiles_pred_topk, canonical_smiles_true))
            mols_results = mols_future.result()
            with tqdm(total=len(canonical_smiles_true), desc="Calculate valid tanimoto: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        max_tani = np.max(result)
                        tanimoto_sims.append(max_tani)
                        all_tanimoto_sims.append(result)
                    except StopIteration:
                        break
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        if save_path:
            np.save(os.path.join(save_path, "tanimoto.npy"), np.array(all_tanimoto_sims))
        metric_vals["topk_tanimoto_sim"] =float(np.mean(tanimoto_sims))
        
        mces_dists = []
        all_mces = []
        if self.mces:
            with ProcessPool(max_workers=cpu_count) as pool:
                mces_future = pool.map(self.calculate_mces, zip(canonical_smiles_pred_topk, canonical_smiles_true))
                mces_results = mces_future.result()
                with tqdm(total=len(canonical_smiles_true), desc="Calculate MCES: ") as progress_bar:
                    while True:
                        try:
                            topk_mces, keys = next(mces_results)
                            mces_dists.append(np.min(topk_mces))
                            all_mces.append(topk_mces)
                            for key, mce_val in zip(keys, topk_mces):
                                if key not in self.mces_cache:
                                    self.mces_cache[key] = mce_val
                        except StopIteration:
                            break
                        except Exception as error:
                            print(error, flush=True)
                            exit()
                        progress_bar.update(1)
            metric_vals["topk_mces_dist"] = float(np.mean(mces_dists))
            if save_path: 
                np.save(os.path.join(save_path, "mces.npy"), np.array(all_mces))

        correct_cnt = 0
        for smiles_topk, smiles_true in zip(canonical_smiles_pred_topk, canonical_smiles_true):
            if smiles_true in smiles_topk:
                correct_cnt += 1
        metric_vals["topk_mol_accuracy"] = correct_cnt / len(canonical_smiles_true)
        for key in metric_vals:
            metric_vals[key] = round(metric_vals[key], 4)
        
        if return_all:
            return metric_vals, all_tanimoto_sims, all_mces
        else:
            return metric_vals