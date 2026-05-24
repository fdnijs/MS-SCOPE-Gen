import os
import json
import numpy as np
import pandas as pd
import pickle
import subprocess
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

def canonical_smiles(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def _accuracy_matrix(labels, preds, max_size):
    n = len(labels)
    mat = np.zeros((n, max_size), dtype=np.float32)

    for i in range(n):
        ref = canonical_smiles(labels[i])

        for j in range(max_size):
            pred = canonical_smiles(preds[i][j])

            if ref is None or pred is None:
                mat[i, j] = 0.0
            else:
                mat[i, j] = 1.0 if ref == pred else 0.0

    return mat

def _build_fps_from_sequences(labels, preds):
    fps = {}

    def fp(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    # ONLY from labels + preds (your requirement)
    for s in labels:
        if s not in fps:
            fps[s] = fp(s)

    for row in preds:
        for s in row:
            if s not in fps:
                fps[s] = fp(s)

    return fps

def _tanimoto_matrix(labels, preds, max_size):
    fps = _build_fps_from_sequences(labels, preds)

    n = len(labels)
    mat = np.zeros((n, max_size), dtype=np.float32)

    for i in range(n):
        f1 = fps.get(labels[i], None)

        for j in range(max_size):
            f2 = fps.get(preds[i][j], None)

            if f1 is None or f2 is None:
                mat[i, j] = np.nan
            else:
                mat[i, j] = DataStructs.TanimotoSimilarity(f1, f2)

    return mat

def _run_mces_batch(labels, preds, max_size, workdir):
    """
    ONE MCES call for entire dataset.
    """

    os.makedirs(workdir, exist_ok=True)

    input_csv = os.path.join(workdir, "input_pairs.csv")
    output_csv = os.path.join(workdir, "mces_out.csv")
    out_npy = os.path.join(workdir, "mces.npy")

    # -----------------------
    # return cached result
    # -----------------------
    if os.path.exists(out_npy):
        print("[LOAD] MCES cached")
        return np.load(out_npy)

    # -----------------------
    # build FULL dataset (N * max_size)
    # -----------------------
    rows = []
    idx = 0

    for i, lab in enumerate(labels):
        for j in range(max_size):
            rows.append((idx, lab, preds[i][j], i))
            idx += 1

    df = pd.DataFrame(rows, columns=["id", "label", "pred", "anchor"])
    df.to_csv(input_csv, index=False, header=False)

    # -----------------------
    # RUN MCES
    # -----------------------
    cmd = [
        "python3",
        "-m", "myopic_mces.myopic_mces",
        "--threshold", "15",
        "--solver", "CPLEX_CMD",
        "--solver_onethreaded",
        "--solver_no_msg",
        input_csv,
        output_csv
    ]

    print("[RUN] MCES batch solver")
    subprocess.run(cmd, check=True)

    # -----------------------
    # LOAD RESULTS
    # -----------------------
    mces = pd.read_csv(output_csv, header=None)

    scores = mces.iloc[:, 1].to_numpy()

    # sanity check
    expected = len(labels) * max_size
    assert len(scores) == expected, (len(scores), expected)

    # reshape to matrix
    mat = scores.reshape(len(labels), max_size)

    np.save(out_npy, mat)

    return mat

def format_data(data_dir, adm="tani", adm_threshold=0.4, max_size=100):
    sample_path = os.path.join(data_dir, "samples.jsonl")

    all_preds_sorted = []
    all_labels = []

    # --- load predictions ---
    with open(sample_path, "r") as f:
        for line in f:
            item = json.loads(line)
            all_preds_sorted.append(item["pred_sorted"][:max_size])
            all_labels.append(item["label"])
    
    scores = np.load(os.path.join(data_dir, "scores__logprob_sorted.npy"))
    
    if adm == "tani":
        out_path = os.path.join(data_dir, "tanimoto_sorted.npy")

        if os.path.exists(out_path):
            vals = np.load(out_path)
        else:
            vals = _tanimoto_matrix(all_labels, all_preds_sorted, max_size)
            np.save(out_path, vals)

    elif adm == "mces":
        vals = _run_mces_batch(
            all_labels,
            all_preds_sorted,
            max_size=max_size,
            workdir=os.path.join(data_dir, "mces_cache")
        )
    elif adm == "acc":
        out_path = os.path.join(data_dir, "acc_sorted.npy")

        if os.path.exists(out_path):
            vals = np.load(out_path)
        else:
            vals = _accuracy_matrix(all_labels, all_preds_sorted, max_size)
            np.save(out_path, vals)

    else:
        raise ValueError(f"Unknown adm: {adm}")
        
    data = []
    for i, (preds, label_smiles) in enumerate(zip(all_preds_sorted, all_labels)):
        n = len(preds)

        scores_ = scores[i][:n]
        scores_ = np.exp(scores_)

        # --- compute labels ---
        if adm == "tani":
            labels_ = (np.array(vals[i]) >= adm_threshold).astype(np.int8)
        elif adm == "mces":
            labels_ = (np.array(vals[i]) <= adm_threshold).astype(np.int8)
        elif adm == "acc":
            labels_ = (np.array(vals[i]) == 1).astype(np.int8)
        else:
            raise ValueError(f"Unknown adm: {adm}")

        # --- compute similarity matrix ---
        mols = []
        valid_indices = []
        for idx, smi in enumerate(preds):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            else:
                mols.append(mol)
                valid_indices.append(idx)
        
        # --- compute fingerprints ---
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mols]
        
        # --- compute similarity matrix ---
        n = len(fps)
        sim_matrix = np.zeros((len(preds), len(preds)), dtype=np.float16)
        
        for i, fp in enumerate(fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps[i:])
            idx_i = valid_indices[i]
            for j, sim in enumerate(sims):
                idx_j = valid_indices[i + j]
                sim_matrix[idx_i, idx_j] = sim
                sim_matrix[idx_j, idx_i] = sim  # mirror

        instance = {
            "labels": labels_,
            "scores": scores_,
            "similarities": sim_matrix
        }

        data.append(instance)

    # --- save ---
    output_path = os.path.join(data_dir, f"{adm}_{adm_threshold}.pkl")
    if adm == "acc":
        output_path = os.path.join(data_dir, f"{adm}.pkl")

    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

    return data