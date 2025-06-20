from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "DeepSeek-R1-Distill-Qwen-7B"  # e.g., "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token if missing (for some models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
