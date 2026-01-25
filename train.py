import argparse
import os
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from src import load_model_and_processor, load_flores_data

# setup argparser
parser = argparse.ArgumentParser(description="Fine-tune pipeline")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
parser.add_argument("--layers", type=str, default="attention", choices=["all-linear", "attention"], 
                    help="LoRA target layers")
parser.add_argument("--name", type=str, default="default", help="Name for making output subfolder")
args = parser.parse_args()

# either target all layers or only attention layers
if args.layers == "all-linear":
    target_modules = "all-linear"
else:
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# load model and processor
model, processor = load_model_and_processor()

# set padding
processor.tokenizer.padding_side = "right"
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# create huggingface Dataset object for flores dataset
# follows the specific requirements for Gemma prompts
def get_formatted_dataset():
    # get "raw" data
    sources, targets = load_flores_data()

    formatted_data = []
    for s, t in zip(sources, targets):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": s,
                        "source_lang_code": "en",
                        "target_lang_code": "nl",
                    }
                ]
            },
            {
                "role": "assistant",
                "content": t
            }
        ]
        
        # apply template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

train_dataset = get_formatted_dataset()

# mask user prompt to prevent training on it
response_template_ids = processor.tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids, 
    tokenizer=processor.tokenizer
)

# lora config
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank * 2,
    target_modules=target_modules,
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

# trainer args
training_args = TrainingArguments(
    output_dir=os.path.join(os.getcwd(), "outputs", args.name),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=torch.cuda.is_available(),
    logging_steps=5,
    num_train_epochs=2,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused"
    #dataloader_num_workers=4,
)

# create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    formatting_func=lambda x: x["text"],
    max_seq_length=1024,
    data_collator=collator,
    tokenizer=processor.tokenizer,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()