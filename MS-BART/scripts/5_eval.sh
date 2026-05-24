MODEL_PATH=results/finetune-original
TEST_PATH=data/MassSpecGym/MassSpecGym_original_threshold_0.11.tsv

python src/eval/evaluation.py \
    --model_path $MODEL_PATH \
    --test_path $TEST_PATH \
    --num_beams 20 \
    --topk_sampling 10 \
    --temp 0.5

# --compute_mces
# --fps_true
# --use_cache (evaluate earlier predictions)
# --num_beams default=100 (also num_return_seq)
# --topk_sampling default=None, k in topk sampling
# --nucleus_sampling default=None, p in nucleus sampling
# --temp default=1.0
# --topk" default=10, Number of sequences to evaluate
