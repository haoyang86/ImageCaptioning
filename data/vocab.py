import re
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Dict, Union


class Vocabulary:
    """
    Shared vocabulary class for image captioning.
    Used by M1 / M2 / M3 / M4.
    """

    def __init__(self, freq_threshold: int = 5):

        self.freq_threshold = freq_threshold

        # special tokens (must stay fixed)
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        self.itos: Dict[int, str] = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token,
        }

        self.stoi: Dict[str, int] = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }

    @property
    def pad_idx(self):
        return self.stoi[self.pad_token]

    @property
    def start_idx(self):
        return self.stoi[self.start_token]

    @property
    def end_idx(self):
        return self.stoi[self.end_token]

    @property
    def unk_idx(self):
        return self.stoi[self.unk_token]

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text: str) -> List[str]:
        """
        Simple tokenizer:
        lowercase + remove punctuation
        """
        text = text.lower().strip()

        tokens = re.findall(
            r"\b\w+\b",
            text
        )

        return tokens

    def build_vocabulary(
        self,
        sentence_list: List[str]
    ):

        frequencies = Counter()

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        idx = len(self.itos)

        for word, freq in frequencies.items():

            if (
                freq >= self.freq_threshold
                and word not in self.stoi
            ):
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(
        self,
        text: str
    ) -> List[int]:
        """
        Convert caption to ids:
        <start> ... <end>
        """

        tokens = self.tokenizer(text)

        ids = [self.start_idx]

        for token in tokens:
            ids.append(
                self.stoi.get(
                    token,
                    self.unk_idx
                )
            )

        ids.append(self.end_idx)

        return ids

    def decode_ids(
        self,
        token_ids: List[int],
        remove_special_tokens=True
    ) -> str:

        words = []

        special_tokens = {
            self.pad_token,
            self.start_token,
            self.end_token
        }

        for idx in token_ids:

            word = self.itos.get(
                int(idx),
                self.unk_token
            )

            if remove_special_tokens and word in special_tokens:
                continue

            words.append(word)

        return " ".join(words)

    def save(
        self,
        path: Union[str, Path]
    ):
        """
        Save vocabulary to a pickle file.

        This saves the exact word-index mapping so that all models
        can share the same vocabulary.
        """

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        vocab_data = {
            "freq_threshold": self.freq_threshold,
            "stoi": self.stoi,
            "itos": self.itos,
            "pad_token": self.pad_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "unk_token": self.unk_token,
        }

        with open(path, "wb") as f:
            pickle.dump(
                vocab_data,
                f
            )

    @classmethod
    def load(
        cls,
        path: Union[str, Path]
    ):
        """
        Load vocabulary from a pickle file.
        """

        path = Path(path)

        with open(path, "rb") as f:
            vocab_data = pickle.load(f)

        vocab = cls(
            freq_threshold=vocab_data["freq_threshold"]
        )

        vocab.stoi = vocab_data["stoi"]
        vocab.itos = vocab_data["itos"]

        vocab.pad_token = vocab_data.get("pad_token", "<pad>")
        vocab.start_token = vocab_data.get("start_token", "<start>")
        vocab.end_token = vocab_data.get("end_token", "<end>")
        vocab.unk_token = vocab_data.get("unk_token", "<unk>")

        return vocab