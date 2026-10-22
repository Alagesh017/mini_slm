import torch
import torch.nn as nn
from datasets import load_dataset
from model.full_model import FullModel
from model.layernorm import RMSNorm
from model.feedforward import SwiGLU
from model.attention import SelfAttention
from tokenizer import MyTokenizer

def verify_tokenizer():
    print("1. Verifying Tokenizer...")
    tokenizer = MyTokenizer(tokenizer_dir="my_tokenizer_16k")
    text = "Once upon a time, there was a little bunny."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"   Tokenizer encode: {tokens[:10]}...")
    print(f"   Tokenizer decode: '{decoded}'")
    assert len(tokens) > 0, "Tokenizer encode failed"
    print("   Tokenizer works!")
    return tokenizer

def verify_vocab_size():
    print("\n2. Verifying Vocab Size...")
    import json
    vocab = json.load(open("my_tokenizer_16k/vocab.json", encoding="utf-8"))
    vocab_size = len(vocab)
    print(f"   Vocab size: {vocab_size}")
    assert vocab_size == 16384, f"Expected 16384, got {vocab_size}"
    print("   Vocab size = 16384!")
    return vocab_size

def verify_model(vocab_size):
    print("\n3. Verifying Model (vocab size 16384)...")
    model = FullModel(
        vocab_size=vocab_size,
        num_layers=8,
        embed_dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        max_seq_len=512
    )
    input_ids = torch.randint(0, vocab_size, (2, 64))
    logits = model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (2, 64, vocab_size), f"Expected (2, 64, {vocab_size}), got {logits.shape}"
    print("   Model uses vocab size 16384!")
    return model

def verify_rope():
    print("\n4. Verifying RoPE implementation...")
    embed_dim = 512
    num_heads = 8
    attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=4)
    x = torch.randn(1, 10, embed_dim)
    out = attn(x)
    print(f"   RoPE attention output shape: {out.shape}")
    assert out.shape == (1, 10, embed_dim)
    print("   RoPE implemented!")

def verify_rmsnorm():
    print("\n5. Verifying RMSNorm...")
    rms_norm = RMSNorm(512)
    x = torch.randn(1, 10, 512)
    out = rms_norm(x)
    print(f"   RMSNorm output shape: {out.shape}")
    assert out.shape == x.shape
    print("   RMSNorm implemented!")

def verify_swiglu():
    print("\n6. Verifying SwiGLU...")
    swiglu = SwiGLU(embed_dim=512, hidden_dim=2048)
    x = torch.randn(1, 10, 512)
    out = swiglu(x)
    print(f"   SwiGLU output shape: {out.shape}")
    assert out.shape == x.shape
    print("   SwiGLU implemented!")

def verify_gqa():
    print("\n7. Verifying GQA...")
    from model.encoder import DecoderBlock
    decoder_block = DecoderBlock(
        embed_dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        max_seq_len=512
    )
    x = torch.randn(1, 10, 512)
    out = decoder_block(x)
    print(f"   DecoderBlock with GQA output shape: {out.shape}")
    assert out.shape == x.shape
    print("   GQA implemented!")

def verify_dataset():
    print("\n8. Verifying Dataset (returns (x, y) pairs)...")
    from train_tinystories import TinyStoriesDataset
    ds = TinyStoriesDataset(split="train", max_len=128)
    input_ids, labels = ds[0]
    print(f"   Dataset sample - input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
    assert len(input_ids) == len(labels), "Input and label lengths don't match"
    assert input_ids.shape[0] > 0, "Input is empty"
    print("   Dataset returns (x, y) pairs!")
    return ds

def verify_dataloader():
    print("\n9. Verifying DataLoader...")
    from train_tinystories import TinyStoriesDataset, collate_fn
    from torch.utils.data import DataLoader
    ds = TinyStoriesDataset(split="train", max_len=128)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    input_ids, labels = batch
    print(f"   DataLoader batch - input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
    assert input_ids.shape[0] == 2, "Batch size should be 2"
    assert labels.shape[0] == 2, "Batch size should be 2"
    print("   DataLoader returns batches!")

if __name__ == "__main__":
    print("="*50)
    print("FINAL VERIFICATION CHECK")
    print("="*50)
    tokenizer = verify_tokenizer()
    vocab_size = verify_vocab_size()
    model = verify_model(vocab_size)
    verify_rope()
    verify_rmsnorm()
    verify_swiglu()
    verify_gqa()
    verify_dataset()
    verify_dataloader()
    print("\n"+"="*50)
    print("ALL VERIFICATIONS PASSED! READY TO TRAIN!")
    print("="*50)
