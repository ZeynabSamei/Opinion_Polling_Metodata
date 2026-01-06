import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # or your base model
TRAIN_FILE = "dpo_vote_2024.jsonl"
OUTPUT_DIR = "./dpo_vote_ft"

BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-4
MAX_LENGTH = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load tokenizer and model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# -------------------------------
# LoRA config
# -------------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA-style
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model.train()

# -------------------------------
# Load DPO dataset
# -------------------------------
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

def preprocess(example):
    # Merge prompt + preferred response
    prompt_text = example["prompt"]
    preferred_text = example["preferred_response"]
    rejected_text = example["rejected_response"]
    
    # For DPO, we need tokenized sequences for both preferred and rejected
    pref_tokens = tokenizer(
        prompt_text + "\nAnswer: " + preferred_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    rej_tokens = tokenizer(
        prompt_text + "\nAnswer: " + rejected_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    return {
        "input_ids_pref": pref_tokens["input_ids"],
        "attention_mask_pref": pref_tokens["attention_mask"],
        "input_ids_rej": rej_tokens["input_ids"],
        "attention_mask_rej": rej_tokens["attention_mask"]
    }

dataset = dataset.map(preprocess)

# -------------------------------
# Collate function for DPO
# -------------------------------
def collate_fn(batch):
    input_ids_pref = torch.tensor([x["input_ids_pref"] for x in batch], dtype=torch.long)
    attention_mask_pref = torch.tensor([x["attention_mask_pref"] for x in batch], dtype=torch.long)
    input_ids_rej = torch.tensor([x["input_ids_rej"] for x in batch], dtype=torch.long)
    attention_mask_rej = torch.tensor([x["attention_mask_rej"] for x in batch], dtype=torch.long)
    return {
        "input_ids_pref": input_ids_pref,
        "attention_mask_pref": attention_mask_pref,
        "input_ids_rej": input_ids_rej,
        "attention_mask_rej": attention_mask_rej
    }

# -------------------------------
# DPO loss function
# -------------------------------
def dpo_loss(model, batch, reference_model, beta=0.1):
    # Forward pass for preferred
    outputs_pref = model(input_ids=batch["input_ids_pref"], attention_mask=batch["attention_mask_pref"], labels=batch["input_ids_pref"])
    logp_pref = -outputs_pref.loss  # negative loss = log-likelihood

    # Forward pass for rejected
    outputs_rej = model(input_ids=batch["input_ids_rej"], attention_mask=batch["attention_mask_rej"], labels=batch["input_ids_rej"])
    logp_rej = -outputs_rej.loss

    # Optionally: compare to reference model for regularization
    with torch.no_grad():
        ref_pref = reference_model(input_ids=batch["input_ids_pref"], attention_mask=batch["attention_mask_pref"], labels=batch["input_ids_pref"])
        ref_rej = reference_model(input_ids=batch["input_ids_rej"], attention_mask=batch["attention_mask_rej"], labels=batch["input_ids_rej"])
        logp_ref = -(ref_pref.loss - ref_rej.loss)

    # DPO loss: negative log-sigmoid of log-prob difference
    loss = -torch.nn.functional.logsigmoid(beta * (logp_pref - logp_rej)).mean()
    return loss

# -------------------------------
# Trainer wrapper
# -------------------------------
from torch.utils.data import DataLoader

reference_model = base_model.eval()  # freeze reference

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss = dpo_loss(model, batch, reference_model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# -------------------------------
# Save the fine-tuned DPO model
# -------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
