from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser
)
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    TrlParser,
    ModelConfig,
    SFTConfig,
    get_peft_config,
    ScriptArguments,
)
from peft import LoraConfig
import os

os.environ["WANDB_LOG_MODEL"] = "end"

def format_instruction_dolly(sample):
    outputs = []
    for i in range(len(sample['instruction'])):
        if "context" in sample and sample["context"][i]:
            text = "Context: " + sample["context"][i] + "\nUser: " + sample["instruction"][i] +\
                "\nAssistant: " + sample["response"][i]
        else:
            text = "User: " + sample["instruction"][i] + "\nAssistant: " + sample["response"][i]
        outputs.append(text)
    return outputs

def format_instruction_alpaca(sample):
    outputs = []
    for i in range(len(sample['instruction'])):
        if "input" in sample and sample["input"][i]:
            text = "Context: " + sample["input"][i] + "\nUser: " + sample["instruction"][i] +\
                "\nAssistant: " + sample["output"][i]
        else:
            text = "User: " + sample["instruction"][i] + "\nAssistant: " + sample["output"][i]
        outputs.append(text)
    return outputs

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    model_config.attn_implementation = "flash_attention_2"

    # Load model and tokenizer
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map= None,
    )
    training_args.model_init_kwargs = model_kwargs
    training_args.deepspeed="config/zero3.json"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token # this should be different?
    tokenizer.padding_side = "right"

    # Load dataset from the hub
    if "playground" in script_args.dataset_name:
        dataset = load_dataset("json", data_files=script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name)

    response_template = [13, 7900, 22137, 29901] #"\nAssistant:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    max_seq_length = 2048 # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=None,
        processing_class=tokenizer,
        formatting_func=format_instruction_dolly if "dolly" in script_args.dataset_name else format_instruction_alpaca,
        data_collator=collator,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)