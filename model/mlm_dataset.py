import random
import torch
from torch.utils.data import Dataset

from tokenizer import MyTokenizer


class MLMDataset(Dataset):

    def __init__(
        self,
        file_path,
        seq_len=128,
        mask_token_id=4
    ):

        self.tokenizer = MyTokenizer()

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        self.tokens = self.tokenizer.encode(
            text
        )

        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

    def __len__(self):

        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):

        start = idx * self.seq_len
        end = start + self.seq_len

        tokens = self.tokens[start:end]

        input_ids = tokens.copy()

        labels = [-100] * len(tokens)

        for i in range(len(tokens)):

            if random.random() < 0.15:

                labels[i] = tokens[i]

                input_ids[i] = self.mask_token_id

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            torch.tensor(
                labels,
                dtype=torch.long
            )
        )