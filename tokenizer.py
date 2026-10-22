from tokenizers import ByteLevelBPETokenizer
import os


class MyTokenizer:

    def __init__(self, tokenizer_dir="my_tokenizer_16k"):
        # Resolve paths relative to the project root
        project_root = os.path.abspath(os.path.dirname(__file__))
        vocab_path = os.path.join(project_root, tokenizer_dir, "vocab.json")
        merges_path = os.path.join(project_root, tokenizer_dir, "merges.txt")

        self.tokenizer = ByteLevelBPETokenizer(
            vocab_path,
            merges_path
        )

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)