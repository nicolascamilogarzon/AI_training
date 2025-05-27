from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "dataset.txt", "validation": "dataset.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


