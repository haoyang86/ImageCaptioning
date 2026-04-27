import re
from collections import Counter
from typing import List, Dict


class Vocabulary:
    """
    Shared vocabulary class for image captioning.
    Used by M1 / M2 / M3.
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
        simple tokenizer:
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
        convert caption to ids:
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