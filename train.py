import os
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from src import load_model_and_processor, load_flores_data
from src.config import OUTPUT_DIR

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
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# trainer args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    #learning_rate=1e-6,
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    remove_unused_columns=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused"
#    dataloader_num_workers=4,
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