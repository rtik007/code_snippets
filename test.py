# ---- installs assumed OK; imports ----
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------- 1) Model & tokenizer ----------
MODEL_NAME = "meta-llama/Llama-3.2-1B"   # <- change to your checkpoint

# Quantization config (4-bit or 8-bit). Pick ONE.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # or: load_in_8bit=True
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16 if no bf16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # important for CausalLM training
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Optional: gradient checkpointing helps memory with long seqs
model.gradient_checkpointing_enable()

# ---------- 2) Prepare for k-bit LoRA & attach adapters ----------
# This is CRUCIAL when using 4/8-bit. It sets up inputs/ln layers correctly.
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules list works well for LLaMA-family. Adjust for other models if needed.
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # should show ~1–10M trainable params, not 0

# ---------- 3) Data Collator ----------
# Ensures batches have labels = input_ids with padding masked to -100.
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------- 4) Trainer config (new TRL style) ----------
sft_config = SFTConfig(
    output_dir="outputs/sft_llama3",
    dataset_text_field="text",   # <- your dataset column with the training text
    max_seq_length=2048,
    packing=False,               # True to pack multiple samples per sequence

    # TrainingArguments fields:
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,          # LoRA on small models can use higher LR; tune as needed
    warmup_ratio=0.03,
    weight_decay=0.0,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    lr_scheduler_type="cosine",
    bf16=True,                   # or fp16=True if no bf16
    report_to="none",
)

# ---------- 5) Build trainer ----------
# dataset_dict must be a HF Datasets dict with "train" and (optionally) "test" splits.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict.get("test"),
    data_collator=collator,      # guarantees `labels` are present
)

# ---------- 6) Train ----------
trainer.train()

# ---------- 7) Save adapter ----------
# (Saves only LoRA weights; base model remains separate.)
trainer.save_model("outputs/sft_llama3_lora")
tokenizer.save_pretrained("outputs/sft_llama3_lora")

##################
# If your data is prompt + response
# Replace dataset_text_field="text" with a formatting function so TRL can turn each row into a single string:

def join_prompt_response(example):
    # craft the final training text per row (add separators/system tags if you use them)
    return f"{example['prompt']}{example['response']}"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict.get("test"),
    formatting_func=join_prompt_response,  # <-- instead of dataset_text_field
    data_collator=collator,
)
