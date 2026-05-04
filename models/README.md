# Run commands

## Optuna tuning

```bash
# uv (preferred)
uv run opt_captioning.py m1
uv run opt_captioning.py m4

# python (venv active)
python opt_captioning.py m1
python opt_captioning.py m4
```

Optional positional args: `[data_pct] [n_trials] [epochs]`. Defaults: `0.5 25 15`.

Trials accumulate in `optuna_studies/{m1,m4}_optuna.db` (SQLite); rerun the
same command to append trials to an existing study. Final CSV export lands
at `m{1,4}_optuna_trials.csv`.

To seed an existing study from a CSV (one-time, e.g. when migrating from
in-memory runs):

```bash
uv run seed_optuna_from_csv.py m1_optuna_trials.csv m1_optuna
```

## Training

```bash
# uv
uv run training/m1_train.py
uv run training/m4_train.py

# uv module style
uv run python -m training.m1_train
uv run python -m training.m4_train

# python (venv active)
python training/m1_train.py
python -m training.m1_train
```

Reads hyperparameters from `configs/m{1,4}_config.py`. Saves to
`outputs/checkpoints/m{1,4}_{embedding}_{best,last}.pt`.

## Evaluation

```bash
# m1 (greedy is default)
uv run python -m evaluation.m1_eval
uv run python -m evaluation.m1_eval beam            # beam, size=3, lp=0.7
uv run python -m evaluation.m1_eval beam 5 0.6      # custom size + length_penalty

# m4 (same CLI shape)
uv run python -m evaluation.m4_eval
uv run python -m evaluation.m4_eval beam
uv run python -m evaluation.m4_eval beam 5 0.6

# python (venv active)
python -m evaluation.m1_eval
python -m evaluation.m4_eval
```

Predictions and metrics land at
`outputs/predictions/M{1,4}/{experiment}/{val_predictions.json,metrics_results.json}`.

## Embedding strategies

Each model supports three options via `embedding_type` in
`configs/m{1,4}_config.py`:

```python
embedding_type = "random"               # trainable, init from N(0, 0.02); no GloVe needed
embedding_type = "pretrained_frozen"    # GloVe 300d, frozen
embedding_type = "pretrained_finetune"  # GloVe 300d, trainable
```

The `experiment_name` (and therefore checkpoint / prediction filenames) is
derived from `embedding_type`, so the three runs don't collide.

For the two GloVe strategies, download
[`glove.6B.300d.txt`](https://nlp.stanford.edu/data/glove.6B.zip) (extract
`glove.6B.300d.txt` from the zip) and place it at
`datasets/embeddings/glove.6B.300d.txt`. Then re-run training + eval as
normal. The first epoch will load GloVe (~1 GB read) and build a
vocab-aligned matrix; subsequent epochs are unaffected.

Workflow per strategy:

```bash
# 1. edit configs/m1_config.py: embedding_type = "pretrained_frozen"
uv run training/m1_train.py
uv run python -m evaluation.m1_eval beam
# checkpoint: outputs/checkpoints/m1_glove_frozen_best.pt
# metrics:    outputs/predictions/M1/m1_glove_frozen/metrics_results.json

# 2. flip to "pretrained_finetune", repeat
# 3. flip to "random", repeat (already done)
```
