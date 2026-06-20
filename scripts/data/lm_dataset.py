"""LM dataset: tokenise processed text for MentalRoBERTa fine-tuning.

Used by scripts/features/semantic/finetune_mental_roberta.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import config


class MentalHealthDataset(Dataset):
    """Returns tokenised text and its integer class label."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.texts = df[config.TEXT_COL].fillna("").astype(str).tolist()
        self.labels = df[config.LABEL_COL].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def processed_split_path(split: str) -> Path:
    """Return the processed CSV path for a split."""
    path_map = {
        "train": config.TRAIN_PATH,
        "val": config.VAL_PATH,
        "test": config.TEST_PATH,
    }
    if split not in path_map:
        raise ValueError("split must be one of: train, val, test")
    return path_map[split]


def load_processed_split(split: str) -> pd.DataFrame:
    """Load one processed split for LM fine-tuning/evaluation."""
    return pd.read_csv(processed_split_path(split))


def build_lm_dataset(
    split: str,
    tokenizer,
    max_length: int = config.MAX_LENGTH,
) -> MentalHealthDataset:
    """Build a tokenising dataset for a processed split."""
    return MentalHealthDataset(
        load_processed_split(split),
        tokenizer=tokenizer,
        max_length=max_length,
    )


def build_lm_dataloader(
    split: str,
    tokenizer,
    max_length: int = config.MAX_LENGTH,
    batch_size: int = config.ROBERTA_BATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader ready for MentalRoBERTa training or evaluation."""
    dataset = build_lm_dataset(split, tokenizer, max_length=max_length)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_lm_dataloaders(
    tokenizer,
    splits: Iterable[str] = ("train", "val", "test"),
    max_length: int = config.MAX_LENGTH,
    batch_size: int = config.ROBERTA_BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Build DataLoaders for multiple processed splits."""
    return {
        split: build_lm_dataloader(
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split in splits
    }
