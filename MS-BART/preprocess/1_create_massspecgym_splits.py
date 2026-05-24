import os
import pandas as pd
import numpy as np


LABELS_FILE = "data/MassSpecGym/mist/labels.tsv"
OUT_DIR = "data/MassSpecGym/mist/splits"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, sep="\t", index=False)
    print(f"[OK] saved {path}")


def mces_split(df):
    keys = df["inchikey"].unique()

    np.random.seed(0)
    np.random.shuffle(keys)

    n = len(keys)
    split = {}

    for k in keys[:int(0.8 * n)]:
        split[k] = "train"
    for k in keys[int(0.8 * n):int(0.9 * n)]:
        split[k] = "val"
    for k in keys[int(0.9 * n):]:
        split[k] = "test"

    out = pd.DataFrame({
        "name": df["spec"],
        "split": df["inchikey"].map(split)
    })

    save(out, f"{OUT_DIR}/mces1.tsv")


def random_split(df):
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    df["split"] = "train"
    df.loc[train_end:val_end, "split"] = "val"
    df.loc[val_end:, "split"] = "test"

    out = pd.DataFrame({
        "name": df["spec"],
        "split": df["split"]
    })

    save(out, f"{OUT_DIR}/random.tsv")

    # optional leakage check (same logic you had)
    train_keys = set(df[df["split"] == "train"]["inchikey"])
    val_keys = set(df[df["split"] == "val"]["inchikey"])
    test_keys = set(df[df["split"] == "test"]["inchikey"])

    print("Unique InChIKeys:")
    print(len(train_keys), len(val_keys), len(test_keys))

    print("Overlap with train:")
    print("val:", len(val_keys & train_keys))
    print("test:", len(test_keys & train_keys))


def main():
    df = pd.read_csv(LABELS_FILE, sep="\t")

    assert "spec" in df.columns
    assert "inchikey" in df.columns

    mces_split(df)
    random_split(df)


if __name__ == "__main__":
    main()