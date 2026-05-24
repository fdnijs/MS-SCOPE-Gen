TOKENIZER_NAME=data/tokenizer
PRETRAIN_FILE=data/MassSpecGym/molecules/processed_molecules.tsv
SAVE_NAME=pretrained-model
LOGGING_STEPS=500
EPOCHS=15
NUM_PROS=1

python src/pretrain/main_trainer.py \
    --tokenizer_name $TOKENIZER_NAME \
    --do_train \
    --do_eval \
    --pretrain_path $PRETRAIN_FILE \
    --preprocessing_num_workers $NUM_PROS \
    --log_level debug \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr 3e-4 \
    --warmup_steps 10000 \
    --num_train_epochs $EPOCHS \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir results/$SAVE_NAME \
    --max_seq_length 256 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --load_best_model_at_end \
    --torch_dtype bfloat16
