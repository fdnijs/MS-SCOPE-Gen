import pandas as pd


def merge(main_path: str, split_path: str):
    print(f"[INFO] Merging:\n  main={main_path}\n  split={split_path}")

    df = pd.read_csv(main_path, sep="\t")
    sp = pd.read_csv(split_path, sep="\t")

    if "identifier" not in df.columns:
        raise ValueError(f"Missing 'identifier' in {main_path}")

    if not {"name", "split"}.issubset(sp.columns):
        raise ValueError(f"{split_path} must contain columns ['name','split']")

    df = df.merge(
        sp[["name", "split"]],
        how="left",
        left_on="identifier",
        right_on="name"
    )

    df["fold"] = df["split"]

    missing = df["fold"].isna().sum()
    print(f"[INFO] Unmatched rows: {missing}")

    df = df.drop(columns=["name", "split"])

    df.to_csv(main_path, sep="\t", index=False)
    print(f"[OK] saved {main_path}")


def main():
    # MCES1
    merge(
        "data/MassSpecGym/MassSpecGym_mces1_threshold_0.11.tsv",
        "data/MassSpecGym/mist/splits/mces1.tsv"
    )

    # RANDOM
    merge(
        "data/MassSpecGym/MassSpecGym_random_threshold_0.11.tsv",
        "data/MassSpecGym/mist/splits/random.tsv"
    )


if __name__ == "__main__":
    main()