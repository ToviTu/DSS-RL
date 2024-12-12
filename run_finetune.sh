DATA=dss

torchrun --nproc_per_node=4 --master_port=25000 finetune.py \
    --seed 42 \
    --deepspeed "./config/zero3.json" \
    --model_name_or_path "meta-llama/llama-2-7b-hf" \
    --output_dir "playground/llama-2-7b-dolly-6k-${DATA}" \
    --dataset_name "playground/dolly_6k_${DATA}.json" \
    --torch_dtype bfloat16 \
    --dataset_train_split "train" \
    --max_seq_length 2048 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --weight_decay 0. \
    --learning_rate 2e-4 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --evaluation_strategy "no" \
    --bf16 True \
    --tf32 True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --attn_implementation "flash_attention_2" \
    --group_by_length True \
    --report_to "wandb" \
    --logging_steps 1 \
