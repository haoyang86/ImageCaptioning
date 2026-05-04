"""
Optuna hyperparameter tuning for image captioning models (M1 and M4).

Tunes one model per study so the search spaces stay clean. Run from the repo root:

    uv run opt_captioning.py <model_id> [data_pct] [n_trials] [epochs]
        model_id : "m1" or "m4"
        data_pct : fraction of training data per trial (default 0.5)
        n_trials : number of optuna trials (default 25)
        epochs   : epochs per trial (default 15)

Examples:
    uv run opt_captioning.py m1
    uv run opt_captioning.py m4 0.5 25 15
    uv run opt_captioning.py m1 0.25 10 8     # quick smoke run

Design notes:
    - batch_size, optimizer, encoder_dim, max_len, freq_threshold are FROZEN.
    - embedding_type is locked at "random" so we don't need GloVe and so the
      embedding strategy stays a study variable, not an HP.
    - Vocabulary is built on the FULL train set, then we Subset the train
      indices. Vocab size stays constant across trials (so val_loss is
      directly comparable), only the data the model sees per epoch changes.
    - Validation always runs on FULL val set, never subsampled.
    - PatientPruner(MedianPruner(...)) -- the script prints val_loss every
      epoch and the trial-state breakdown at the end, so you can verify
      pruning actually fired.

Reference:
    "Optimal hyperparameters discovered through small-scale tuning can
    transfer to larger models / longer training" -- Complete(d)P paper,
    arxiv 2512.22382. Practical takeaway: when going from these tuned
    HPs at <epochs> epochs to a longer final run at K * <epochs>, divide
    the tuned LR by approximately sqrt(K).

I referenced my optuna_trials.py from assignments 2 + 3 (CS7643) and the
optuna pytorch examples (https://github.com/optuna/optuna-examples/tree/main/pytorch)
while putting this together.
"""

import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate
from models.model_m1 import ModelM1
from models.model_m4 import ModelM4


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# usage: uv run opt_captioning.py <model_id> [data_pct] [n_trials] [epochs]
MODEL_ID = sys.argv[1].lower() if len(sys.argv) > 1 else "m1"
DATA_PCT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
N_TRIALS = int(sys.argv[3])  if len(sys.argv) > 3 else 25
EPOCHS   = int(sys.argv[4])  if len(sys.argv) > 4 else 15

assert MODEL_ID in {"m1", "m4"}, f"model_id must be m1 or m4, got {MODEL_ID}"
assert 0 < DATA_PCT <= 1.0,      f"data_pct must be in (0, 1], got {DATA_PCT}"

STUDY_NAME = f"{MODEL_ID}_optuna"
print(f"Running optuna study: {STUDY_NAME}")
print(f"  model_id={MODEL_ID}, data_pct={DATA_PCT}, n_trials={N_TRIALS}, epochs={EPOCHS}")


# --------------------------------------------------------------------------
# Constants we are NOT tuning
# --------------------------------------------------------------------------

BATCH_SIZE   = 32
ENCODER_DIM  = 512
MAX_LEN      = 50
EMBED_DIM    = 300        # consistent with cfg.embedding_dim, even with random embeddings
FREQ_THRESH  = 5
GRAD_CLIP    = 1.0
SUBSET_SEED  = 6739       # same seed I've used in past CS7643 trials, locks the train subset

DATASET_NAME = "flickr8k"
IMAGE_FOLDER = "Flicker8k_Dataset"
IMAGE_SIZE   = 224

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

# Adding support for mps as I have a macbook pro, but I think my teammates have CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# --------------------------------------------------------------------------
# Data setup (built once, reused across trials)
# --------------------------------------------------------------------------

dataset_root = ROOT / "datasets" / DATASET_NAME
image_dir = dataset_root / IMAGE_FOLDER
train_csv = dataset_root / "train.csv"
val_csv   = dataset_root / "val.csv"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Vocab is built from the FULL train set so it stays stable across trials and
# across data_pct values. Loss is comparable because vocab_size is constant.
train_dataset_full = FlickrDataset(
    image_dir=image_dir,
    caption_file=train_csv,
    transform=transform,
    freq_threshold=FREQ_THRESH,
)
val_dataset = FlickrDataset(
    image_dir=image_dir,
    caption_file=val_csv,
    transform=transform,
    freq_threshold=FREQ_THRESH,
)
val_dataset.vocab = train_dataset_full.vocab

vocab = train_dataset_full.vocab
VOCAB_SIZE = len(vocab)

# Subsample TRAINING indices only. Validation stays full.
np.random.seed(SUBSET_SEED)
n_subset = int(DATA_PCT * len(train_dataset_full))
subset_indices = np.random.choice(len(train_dataset_full), n_subset, replace=False)
train_subset = Subset(train_dataset_full, subset_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=FlickrCollate(pad_idx=vocab.pad_idx),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=FlickrCollate(pad_idx=vocab.pad_idx),
)

print(f"Vocab size: {VOCAB_SIZE}")
print(f"Train samples used per trial: {len(train_subset)} of {len(train_dataset_full)}")
print(f"Val samples (always full):    {len(val_dataset)}")


# --------------------------------------------------------------------------
# Per-model HP suggestion + model build
# --------------------------------------------------------------------------

def suggest_and_build_m1(trial):
    """
    Suggest M1 hyperparameters and build the model.
    """
    lr               = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dropout          = trial.suggest_float("dropout", 0.05, 0.5)
    weight_decay     = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    encoder_dropout  = trial.suggest_float("encoder_dropout", 0.1, 0.3)
    hidden_dim       = trial.suggest_categorical("hidden_dim", [256, 384, 512, 768, 1024])
    num_lstm_layers  = trial.suggest_categorical("num_lstm_layers", [1, 2])

    model = ModelM1(
        vocab_size=VOCAB_SIZE,
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        encoder_dim=ENCODER_DIM,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        max_len=MAX_LEN,
        freeze_backbone=True,
        pretrained=True,
        encoder_dropout=encoder_dropout,
        embedding_type="random",
        embedding_dim=EMBED_DIM,
        pretrained_embedding_matrix=None,
    ).to(DEVICE)

    return model, lr, weight_decay


def suggest_and_build_m4(trial):
    """
    Suggest M4 hyperparameters and build the model. d_model and nhead are
    suggested independently from sets where every (d, h) combo divides cleanly.
    """
    lr               = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dropout          = trial.suggest_float("dropout", 0.05, 0.5)
    weight_decay     = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    encoder_dropout  = trial.suggest_float("encoder_dropout", 0.1, 0.3)
    d_model          = trial.suggest_categorical("d_model", [256, 384, 512, 768])
    nhead            = trial.suggest_categorical("nhead", [4, 8, 16])
    num_layers       = trial.suggest_categorical("num_decoder_layers", [2, 4, 6])
    dim_feedforward  = trial.suggest_categorical("dim_feedforward", [1024, 2048, 4096])

    # All combos of (d_model, nhead) above already divide cleanly; no skip needed.

    model = ModelM4(
        vocab_size=VOCAB_SIZE,
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        encoder_dim=ENCODER_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=MAX_LEN,
        freeze_backbone=True,
        pretrained=True,
        encoder_dropout=encoder_dropout,
        embedding_type="random",
        embedding_dim=EMBED_DIM,
        pretrained_embedding_matrix=None,
    ).to(DEVICE)

    return model, lr, weight_decay


SUGGESTERS = {
    "m1": suggest_and_build_m1,
    "m4": suggest_and_build_m4,
}


# --------------------------------------------------------------------------
# Train / validate (inline so we don't depend on m{1,4}_train.py)
# --------------------------------------------------------------------------

def train_one_epoch(model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, captions in train_loader:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        logits = model(images, decoder_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, criterion):
    model.eval()
    total_loss = 0.0
    for images, captions in val_loader:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        logits = model(images, decoder_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(val_loader)


# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------

def objective(trial):
    torch.manual_seed(SUBSET_SEED)
    np.random.seed(SUBSET_SEED)

    model, lr, weight_decay = SUGGESTERS[MODEL_ID](trial)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(model, criterion, optimizer)
        val_loss   = validate(model, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Print every epoch so we can SEE that reports are reaching optuna.
        # If pruning silently breaks, this trace is the easiest way to spot it.
        print(
            f"  trial {trial.number:>3d} | epoch {epoch:>2d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | best {best_val_loss:.4f}"
        )

        # Tell the pruner what we just saw and let it decide.
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            print(f"  trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    elapsed = time.time() - t0
    print(f"  trial {trial.number} done in {elapsed/60:.1f} min, best val_loss {best_val_loss:.4f}")

    # Free memory between trials. MPS doesn't have empty_cache, so guard it.
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss


# --------------------------------------------------------------------------
# Run study
# --------------------------------------------------------------------------

# Persistent SQLite storage so future runs of this script append to the same
# study instead of starting from scratch. TPE uses all prior trials when
# proposing new ones, so each subsequent batch leverages the history.
storage_dir = ROOT / "optuna_studies"
storage_dir.mkdir(exist_ok=True)
storage = f"sqlite:///{storage_dir / f'{STUDY_NAME}.db'}"

study = optuna.create_study(
    storage=storage,
    study_name=STUDY_NAME,
    direction="minimize",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SUBSET_SEED),
    pruner=optuna.pruners.PatientPruner(
        # n_startup_trials=5 lets the first 5 trials run to completion (default)
        # n_warmup_steps=2 skips pruning on the noisy first two epochs
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        patience=3,
    ),
)

n_existing = len(study.trials)
if n_existing > 0:
    best = study.best_trial.value if study.best_trial.value is not None else None
    best_str = f"{best:.4f}" if best is not None else "n/a"
    print(f"Resuming study {STUDY_NAME} with {n_existing} prior trials "
          f"(best so far: {best_str}). About to add {N_TRIALS} new trials.")

study.optimize(objective, n_trials=N_TRIALS)


# --------------------------------------------------------------------------
# Save trials + sanity check that pruning actually engaged
# --------------------------------------------------------------------------

df = study.trials_dataframe()
csv_path = ROOT / f"{STUDY_NAME}_trials.csv"
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"Study: {STUDY_NAME}")
print(f"  trials saved to: {csv_path}")
print(f"  trial states:")
print(df["state"].value_counts().to_string())
n_pruned = (df["state"] == "PRUNED").sum()
if n_pruned > 0:
    print(f"  pruning engaged on {n_pruned} of {N_TRIALS} trials")
elif N_TRIALS <= 5:
    print(f"  pruning did not engage (expected; n_trials={N_TRIALS} <= MedianPruner startup of 5)")
else:
    print(f"  WARNING: 0 trials were pruned across {N_TRIALS} trials. PatientPruner may not be engaging.")
    print("           Check that the per-epoch val_loss prints appeared in the trace above.")

print(f"\nBest trial:")
print(f"  val_loss: {study.best_trial.value:.4f}")
print(f"  params:")
for k, v in study.best_trial.params.items():
    print(f"    {k}: {v}")

print(f"\nWhen scaling these HPs to a longer final run of K * {EPOCHS} epochs,")
print(f"divide the tuned LR by approximately sqrt(K) (per the Complete(d)P paper).")
print(f"{'='*60}")
