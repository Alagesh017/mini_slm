import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Download TinyStories
print("Loading TinyStories dataset...")
ds = load_dataset("roneneldan/TinyStories")

# Save text to training file
training_file = "tinystories_training_text.txt"
print(f"Saving training text to {training_file}...")

with open(training_file, "w", encoding="utf-8") as f:
    for split in ds:
        for example in ds[split]:
            text = example["text"]
            if text:
                f.write(text + "\n")

print("Training text saved!")

# Train tokenizer with 16k vocab
tokenizer_dir = "my_tokenizer_16k"
os.makedirs(tokenizer_dir, exist_ok=True)

print("Training 16k vocab tokenizer...")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[training_file],
    vocab_size=16384,  # 16k
    min_frequency=2,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
)

tokenizer.save_model(tokenizer_dir)
print(f"Tokenizer trained and saved to {tokenizer_dir}!")
