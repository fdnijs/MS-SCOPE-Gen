import argparse, csv, random
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

def get_morgan_4096(smiles: str, nbits: int = 4096, radius: int = 2):
    if not smiles:
        return ""
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        ones = np.where(arr == 1)[0].tolist()
        return "".join(f"<fp{idx:04d}>" for idx in ones)
    except Exception:
        return ""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="data/MassSpecGym/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv", help="Input TSV (must have a 'smiles' column).")
    p.add_argument("output", nargs="?", default="data/MassSpecGym/molecules/processed_molecules.tsv", help="Output TSV.")
    p.add_argument("--val-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # count rows (expects TSV with header containing "smiles")
    with open(args.input, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames is None or "smiles" not in r.fieldnames:
            raise SystemExit("Input must have a 'smiles' column")
        total = sum(1 for _ in r)

    val_n = min(args.val_size, total)
    rng = random.Random(args.seed)
    val_idx = set(rng.sample(range(total), val_n)) if val_n > 0 else set()

    # stream-process and write TSV: canonical_smiles, fps, split
    with open(args.input, newline="") as inf, open(args.output, "w", newline="") as outf:
        reader = csv.DictReader(inf, delimiter="\t")
        writer = csv.writer(outf, delimiter="\t")
        writer.writerow(["canonical_smiles", "fps", "split"])
        for i, row in enumerate(reader):
            smi = row.get("smiles", "") or ""
            can = ""
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    Chem.RemoveStereochemistry(mol)
                    can = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                can = ""
            fps = get_morgan_4096(can)
            split = "val" if i in val_idx else "train"
            writer.writerow([can, fps, split])