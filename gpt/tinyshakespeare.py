# https://chatgpt.com/c/68530d56-c928-8004-b83c-813b6f9406a0

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from einops import rearrange   # purely for clarity in the collate fn below

class Vocab:
    def __init__(self, tokens: set[str]):
        self.tokens = sorted(tokens)

        self.tok2idx: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.idx2tok: dict[int, str] = {i: tok for i, tok in enumerate(self.tokens)}
    
    @property
    def size(self):
        return len(self.tokens)

    def encode(self, toks: list[str]):
        return [self.tok2idx[tok] for tok in toks]
    
    def decode(self, idxs: list[int]):
        return [self.idx2tok[i] for i in idxs]


class TinyShakespeareDataset(Dataset):
    """Character-level LM dataset for the Tiny Shakespeare corpus."""

    def __init__(self, file_path: str | Path, seq_len: int = 128) -> None:
        text = Path(file_path).read_text(encoding="utf-8")
        self.vocab = Vocab(set(text))

        self.data = torch.tensor(
            self.vocab.encode(list(text)),
            dtype=torch.long
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Each sample needs seq_len+1 tokens
        return self.data.numel() - self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        chunk = self.data[idx:idx + self.seq_len + 1]
        x, y = chunk[:-1], chunk[1:]
        return x, y


def make_train_val_dataloader(
    ds: TinyShakespeareDataset,
    split: float = 0.9,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    split_idx = int(len(ds) * split)
    train_ds = Subset(ds, list(range(0, split_idx)))
    val_ds = Subset(ds, list(range(split_idx, len(ds))))

    # custom collate for clarity
    def collate(batch):
        xs, ys = zip(*batch)
        x = rearrange(torch.stack(xs), "b l -> b l")
        y = rearrange(torch.stack(ys), "b l -> b l")
        return x, y

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate,
        num_workers=num_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate,
        num_workers=num_workers,
    )

    return {
        "train": train_dl,
        "val": val_dl
    }
