import torch
from model.full_model import FullModel
from tokenizer import MyTokenizer

def generate_text(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.7
):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_tokens = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


if __name__ == "__main__":
    # Hyperparameters (must match training)
    VOCAB_SIZE = 16384
    NUM_LAYERS = 8
    EMBED_DIM = 512
    NUM_HEADS = 8
    NUM_KV_HEADS = 4
    HIDDEN_DIM = 2048
    MAX_SEQ_LEN = 512
    CHECKPOINT_PATH = "tinystories_model_epoch_1.pt"  # Update this after training

    # Load tokenizer and model
    tokenizer = MyTokenizer(tokenizer_dir="my_tokenizer_16k")
    model = FullModel(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=MAX_SEQ_LEN
    )

    # Load checkpoint (if exists)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print("Checkpoint not found. Using untrained model.")

    # Interactive loop
    print("\n=== TinyStories Interactive Generator ===")
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        temp = input("Enter temperature (0.1-2.0, default 0.7): ")
        temp = float(temp) if temp else 0.7
        max_tokens = input("Enter max new tokens (default 100): ")
        max_tokens = int(max_tokens) if max_tokens else 100

        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_tokens, temperature=temp)
        print(f"\n{generated}")
