# --- preprocessing (runs once) ---
python ../preprocess/1_create_massspecgym_splits.py


# --- MCES1 training ---
mkdir -p results/mist_mces1

python mist/src/mist/train_mist.py \
    --cache-featurizers \
    --labels-file 'data/MassSpecGym/mist/labels.tsv' \
    --subform-folder 'data/MassSpecGym/mist/subformulae/default_subformulae/' \
    --spec-folder 'data/MassSpecGym/mist/spec_files/' \
    --magma-folder 'data/MassSpecGym/mist/magma_outputs/magma_tsv/' \
    --fp-names morgan4096 \
    --seed 1 \
    --gpus 1 \
    --batch-size 128 \
    --iterative-preds 'growing' \
    --iterative-loss-weight 0.1 \
    --learning-rate 0.0003 \
    --weight-decay 1e-07 \
    --lr-decay-frac 0.9 \
    --hidden-size 512 \
    --pairwise-featurization \
    --peak-attn-layers 3 \
    --refine-layers 5 \
    --spectra-dropout 0.3 \
    --magma-aux-loss \
    --magma-loss-lambda 4 \
    --split-file 'data/MassSpecGym/mist/splits/mces1.tsv' \
    --form-embedder 'pos-cos' \
    --no-diffs \
    --save-dir results/mist_mces1


mv results/mist_mces1/mces1/best.ckpt \
   data/MassSpecGym/mist/ckpts/mces1.ckpt


# --- random training ---
mkdir -p results/mist_random

python mist/src/mist/train_mist.py \
    --cache-featurizers \
    --labels-file 'data/MassSpecGym/mist/labels.tsv' \
    --subform-folder 'data/MassSpecGym/mist/subformulae/default_subformulae/' \
    --spec-folder 'data/MassSpecGym/mist/spec_files/' \
    --magma-folder 'data/MassSpecGym/mist/magma_outputs/magma_tsv/' \
    --fp-names morgan4096 \
    --seed 1 \
    --gpus 1 \
    --batch-size 128 \
    --iterative-preds 'growing' \
    --iterative-loss-weight 0.1 \
    --learning-rate 0.0003 \
    --weight-decay 1e-07 \
    --lr-decay-frac 0.9 \
    --hidden-size 512 \
    --pairwise-featurization \
    --peak-attn-layers 3 \
    --refine-layers 5 \
    --spectra-dropout 0.3 \
    --magma-aux-loss \
    --magma-loss-lambda 4 \
    --split-file 'data/MassSpecGym/mist/splits/random.tsv' \
    --form-embedder 'pos-cos' \
    --no-diffs \
    --save-dir results/mist_random


mv results/mist_random/random/best.ckpt \
   data/MassSpecGym/mist/ckpts/random.ckpt
