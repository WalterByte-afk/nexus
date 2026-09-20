"""
Data loading utilities for NEXUS-Omega.

Handles text datasets, tokenization, and efficient batching.
Designed to work with minimal dependencies.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
import json
import random


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.

    Takes a list of texts and a tokenizer, handles all the padding and truncation.
    Nothing fancy, just works.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Tokenize everything upfront
        print(f"Tokenizing {len(texts)} texts...")
        self.examples = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)

            # Handle short texts: pad to max_length
            if len(tokens) < max_length:
                # Pad with pad_token_id (0)
                padded = tokens + [0] * (max_length - len(tokens))
                self.examples.append(padded)
            else:
                # Create overlapping chunks for long texts
                for i in range(0, len(tokens) - max_length + 1, stride):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.examples.append(chunk)

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }


class StreamingDataset(Dataset):
    """
    For when the dataset doesn't fit in memory.

    Loads data on-the-fly from disk. Slower but uses way less RAM.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        max_length: int = 512,
        cache_size: int = 1000,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size

        # Count lines
        with open(file_path, 'r', encoding='utf-8') as f:
            self.num_lines = sum(1 for _ in f)

        # Simple LRU cache
        self.cache: Dict[int, torch.Tensor] = {}
        self.cache_order = []

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        # Check cache
        if idx in self.cache:
            return self.cache[idx]

        # Load from disk
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    text = line.strip()
                    break

        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
        )

        item = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }

        # Update cache
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[idx] = item
        self.cache_order.append(idx)

        return item


class SimpleTokenizer:
    """
    Basic tokenizer for testing without external dependencies.

    Real training would use HuggingFace tokenizers, but this works for prototyping.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # Build simple vocab
        self.char_to_id = {chr(i + 65): i + 4 for i in range(26)}  # A-Z
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Convert text to token ids."""
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for char in text.upper():
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens[:kwargs.get('max_length', 999999)]

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
            elif id == self.pad_token_id:
                continue
            elif id == self.bos_token_id:
                continue
            elif id == self.eos_token_id:
                break
            else:
                chars.append('?')
        return ''.join(chars)


def create_dataloader(
    texts: List[str],
    tokenizer: Any,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """
    Quick helper to create a dataloader.

    Just pass your texts and tokenizer, get back a ready-to-use dataloader.
    """
    dataset = TextDataset(texts, tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False,
    )


def load_text_file(path: str) -> List[str]:
    """Load lines from a text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_jsonl(path: str, text_key: str = 'text') -> List[str]:
    """Load texts from a JSONL file."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if text_key in data:
                texts.append(data[text_key])
    return texts


if __name__ == "__main__":
    print("Testing data utilities...")

    # Test simple tokenizer
    tok = SimpleTokenizer(vocab_size=1000)

    text = "Hello world"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Test dataset
    texts = ["Hello world", "Testing the dataset", "Another example text"]
    dataset = TextDataset(texts, tok, max_length=16)

    print(f"\nDataset size: {len(dataset)}")
    print(f"First example: {dataset[0]}")

    # Test dataloader
    loader = create_dataloader(texts, tok, batch_size=2)
    batch = next(iter(loader))
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")

    print("\nAll tests passed!")
