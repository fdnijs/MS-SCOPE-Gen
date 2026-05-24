# =========================
# 1. Pretraining data
# =========================
python preprocess/1_generate_pretrain_data.py


# =========================
# 2. MGF + labels
# =========================
python preprocess/2_generate_mgf_and_labels.py


# =========================
# 3. Fingerprint prediction (MIST)
# =========================
python preprocess/3_predict_fp.py --mist-ckpt original.ckpt   # original datasplit
python preprocess/3_predict_fp.py --mist-ckpt mces1.ckpt      # MCES1 distance split
python preprocess/3_predict_fp.py --mist-ckpt random.ckpt     # random split


# =========================
# 4. Merge splits into MassSpecGym tables (REPLACES python -c)
# =========================
python preprocess/4_merge_splits_into_massspecgym.py


# =========================
# 5. Tokenizer download
# =========================
wget -O data/tokenizer/vocab_no_fp.json \
https://huggingface.co/mikemayuare/SMILYAPE/resolve/main/tokenizer.json


# =========================
# 6. Vocabulary generation
# =========================
python preprocess/5_generate_vocab.py
