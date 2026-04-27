from pathlib import Path
from typing import Any, Tuple

import torch


def load_glove_vectors(
    glove_path,
    embedding_dim: int = 300,
) -> dict:
    """
    Load GloVe vectors from txt file.

    Returns:
        glove_dict:
            word -> torch.Tensor(embedding_dim)
    """

    glove_path = Path(glove_path)

    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {glove_path}"
        )

    glove_dict = {}

    print(f"Loading GloVe from: {glove_path}")

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")

            word = values[0]
            vector_values = values[1:]

            if len(vector_values) != embedding_dim:
                continue

            vector = torch.tensor(
                [float(x) for x in vector_values],
                dtype=torch.float,
            )

            glove_dict[word] = vector

    print(f"Loaded {len(glove_dict)} GloVe vectors.")

    return glove_dict


def build_embedding_matrix(
    vocab: Any,
    glove_path,
    embedding_dim: int = 300,
    random_scale: float = 0.02,
) -> Tuple[torch.Tensor, int, int]:
    """
    Build embedding matrix aligned with our Vocabulary.

    Args:
        vocab:
            Vocabulary object with stoi / pad_idx.

        glove_path:
            Path to glove.6B.300d.txt.

        embedding_dim:
            GloVe dimension.

        random_scale:
            Std for random init of OOV words.

    Returns:
        embedding_matrix:
            Tensor of shape (vocab_size, embedding_dim)

        found_count:
            Number of vocab words found in GloVe.

        oov_count:
            Number of vocab words not found in GloVe.
    """

    glove_dict = load_glove_vectors(
        glove_path=glove_path,
        embedding_dim=embedding_dim,
    )

    vocab_size = len(vocab)

    embedding_matrix = torch.empty(
        vocab_size,
        embedding_dim,
        dtype=torch.float,
    )

    embedding_matrix.normal_(
        mean=0.0,
        std=random_scale,
    )

    # pad token should stay zero
    embedding_matrix[vocab.pad_idx].fill_(0.0)

    found_count = 0
    oov_count = 0

    for word, idx in vocab.stoi.items():

        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]
            found_count += 1
        else:
            oov_count += 1

    print(
        f"GloVe coverage: "
        f"{found_count}/{vocab_size} "
        f"({found_count / vocab_size:.2%})"
    )
    print(f"OOV words: {oov_count}")

    return embedding_matrix, found_count, oov_count


def should_use_pretrained_embedding(
    embedding_type: str,
) -> bool:
    """
    Whether current embedding type needs pretrained vectors.
    """

    return embedding_type in {
        "pretrained_frozen",
        "pretrained_finetune",
    }


def is_embedding_frozen(
    embedding_type: str,
) -> bool:
    """
    Whether pretrained embedding should be frozen.
    """

    if embedding_type == "pretrained_frozen":
        return True

    if embedding_type == "pretrained_finetune":
        return False

    if embedding_type == "random":
        return False

    raise ValueError(
        f"Unsupported embedding_type: {embedding_type}"
    )