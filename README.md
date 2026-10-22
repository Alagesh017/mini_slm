# TinyStories LLM (from scratch)

A small GPT-style decoder-only transformer trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

## Architecture

| Component | Implementation |
| --- | --- |
| Attention | Multi-Head Attention with RoPE + GQA (grouped-query attention) |
| Norm | RMSNorm |
| Feed-forward | SwiGLU |
| Tokenizer | ByteLevel BPE, 16k vocab |
| Head | LM head for next-token prediction |

Default config: `vocab_size=16384`, `num_layers=8`, `embed_dim=512`, `num_heads=8`, `num_kv_heads=4`, `hidden_dim=2048`, `max_seq_len=512`.

## Project layout

```
model/
  attention.py      # RoPE + GQA attention
  embeddings.py     # token embeddings
  encoder.py        # DecoderBlock (attention + FFN + norms)
  feedforward.py    # SwiGLU
  layernorm.py      # RMSNorm
  mlm_head.py       # LM head
  transformer.py    # stack of decoder blocks
  full_model.py     # top-level FullModel
tokenizer.py                       # MyTokenizer wrapper
prepare_tinystories_tokenizer.py   # download data + train 16k BPE tokenizer
train_tinystories.py               # training loop
generate.py                        # single-prompt generation
interactive_generate.py            # interactive REPL generator
verify_gpu.py                      # CUDA sanity check
verify_everything.py               # end-to-end component checks
my_tokenizer_16k/                  # trained tokenizer (vocab.json, merges.txt)
```

## Requirements

- Python 3.10+
- PyTorch (CUDA build recommended, CPU works but training is very slow)

```bash
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm
```

Check that your GPU is visible:

```bash
python verify_gpu.py
```

## How to run

### 1. Prepare the tokenizer (only needed once)

Downloads TinyStories, dumps all text to `tinystories_training_text.txt` (several GB), then trains a 16k BPE tokenizer into `my_tokenizer_16k/`.

```bash
python prepare_tinystories_tokenizer.py
```

> The repo already ships `my_tokenizer_16k/`, so you can skip this unless you want to retrain it. The temp text file can be deleted afterwards.

### 2. Verify everything is wired up

```bash
python verify_everything.py
```

Runs checks on the tokenizer, vocab size, model shapes, RoPE, RMSNorm, SwiGLU, GQA, dataset and dataloader.

### 3. Train

Note: training loads the dataset straight from the hub — the tokenizer must exist first.

```bash
python train_tinystories.py
```

Saves a checkpoint after each epoch: `tinystories_model_epoch_1.pt`, `_2.pt`, ...

On CPU this is extremely slow. Better to run on a GPU machine or a Colab/Kaggle notebook.

### 4. Generate text

Single prompt (edit `CHECKPOINT_PATH` in the script if your checkpoint has a different name):

```bash
python generate.py
```

Interactive mode — asks for a prompt, temperature and token count in a loop:

```bash
python interactive_generate.py
```

Example:

```
Enter your prompt (or 'quit' to exit): Once upon a time
Enter temperature (0.1-2.0, default 0.7): 0.8
Enter max new tokens (default 100): 150
```

If no checkpoint is found, the scripts fall back to an untrained model (output will be gibberish).

## Notes / gotchas

- Hyperparameters in `generate.py` and `interactive_generate.py` must match the ones used in `train_tinystories.py`, otherwise loading the checkpoint fails.
- `temperature` controls randomness: lower = more deterministic, higher = more random.
- `tinystories_training_text.txt` is large and only used by the tokenizer step; it is safe to delete.
