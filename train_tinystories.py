import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from model.full_model import FullModel
from tokenizer import MyTokenizer
from tqdm import tqdm

# Hyperparameters
VOCAB_SIZE = 16384
NUM_LAYERS = 8
EMBED_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4  # you can use 2 or 4
FFN_SIZE = 2048
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
GRADIENT_CLIP = 1.0

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer
tokenizer = MyTokenizer(tokenizer_dir="my_tokenizer_16k")
PAD_TOKEN_ID = tokenizer.tokenizer.token_to_id("[PAD]")

# Dataset class
class TinyStoriesDataset(Dataset):
    def __init__(self, split="train", max_len=MAX_SEQ_LEN):
        self.ds = load_dataset("roneneldan/TinyStories", split=split)
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        tokens = tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # For next token prediction
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels

# Collate function
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences
    input_ids_padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignores in loss
    
    return input_ids_padded.to(device), labels_padded.to(device)

# Model
model = FullModel(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    hidden_dim=FFN_SIZE,
    max_seq_len=MAX_SEQ_LEN
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Data loaders
train_dataset = TinyStoriesDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

print("Starting training...")

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in progress_bar:
        input_ids, labels = batch
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids)
        
        # Shift logits and labels
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        # Optimize
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"tinystories_model_epoch_{epoch+1}.pt")

print("Training complete!")
