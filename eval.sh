MODEL=meta-llama/llama-2-7b-hf
#MODEL=playground/llama-2-7b-dolly-6k-dss
#MODEL=playground/llama-2-7b-dolly-6k-random

lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype="bfloat16",peft=playground/llama-2-7b-alpaca-dss/checkpoint-144 \
    --include_path ./evals \
    --tasks mmlu_my,boolq_my,arc_easy_my,arc_challenge_my,hellaswag_my \
    --device cuda:3 \
    --batch_size 32 \
    --num_fewshot 0 \