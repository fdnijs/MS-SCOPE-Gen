MODEL_PATH=results/finetune-original
TEST_PATH=data/MassSpecGym/MassSpecGym_original_threshold_0.11.tsv

python src/eval/evaluation.py \
    --model_path $MODEL_PATH \
    --test_path $TEST_PATH
    
MODEL_PATH=results/finetune-mces1
TEST_PATH=data/MassSpecGym/MassSpecGym_mces1_threshold_0.11.tsv

python src/eval/evaluation.py \
    --model_path $MODEL_PATH \
    --test_path $TEST_PATH

MODEL_PATH=results/finetune-random
TEST_PATH=data/MassSpecGym/MassSpecGym_random_threshold_0.11.tsv

python src/eval/evaluation.py \
    --model_path $MODEL_PATH \
    --test_path $TEST_PATH
