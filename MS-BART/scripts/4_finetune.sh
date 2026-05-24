MODEL_NAME_OR_PATH=results/pretrained-model
TRAIN_FILE=data/MassSpecGym/MassSpecGym_original_threshold_0.11.tsv
SAVE_NAME=finetune-original
LOGGING_STEPS=500
EPOCHS=10
NUM_PROS=1

python src/finetune/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --do_train \
    --do_eval \
    --train_file $TRAIN_FILE \
    --preprocessing_num_workers $NUM_PROS \
    --learning_rate 1e-4 \
    --warmup_ratio 0.2 \
    --num_train_epochs $EPOCHS \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir results/$SAVE_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 128 \
    --load_best_model_at_end \
    --early_stopping_patience 5 \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --torch_dtype bfloat16

TRAIN_FILE=data/MassSpecGym/MassSpecGym_mces1_threshold_0.11.tsv
SAVE_NAME=finetune-mces1

python src/finetune/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --do_train \
    --do_eval \
    --train_file $TRAIN_FILE \
    --preprocessing_num_workers $NUM_PROS \
    --learning_rate 1e-4 \
    --warmup_ratio 0.2 \
    --num_train_epochs $EPOCHS \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir results/$SAVE_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 128 \
    --load_best_model_at_end \
    --early_stopping_patience 5 \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --torch_dtype bfloat16

TRAIN_FILE=data/MassSpecGym/MassSpecGym_random_threshold_0.11.tsv
SAVE_NAME=finetune-random

python src/finetune/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --do_train \
    --do_eval \
    --train_file $TRAIN_FILE \
    --preprocessing_num_workers $NUM_PROS \
    --learning_rate 1e-4 \
    --warmup_ratio 0.2 \
    --num_train_epochs $EPOCHS \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir results/$SAVE_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 128 \
    --load_best_model_at_end \
    --early_stopping_patience 5 \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --torch_dtype bfloat16
