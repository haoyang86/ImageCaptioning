"""
One-time helper to import an existing m{1,4}_optuna_trials.csv into a fresh
SQLite-backed Optuna study. Run this once BEFORE rerunning opt_captioning.py
if you want future trials to leverage prior results.

Usage:
    uv run seed_optuna_from_csv.py <csv_path> <study_name>

Example:
    uv run seed_optuna_from_csv.py m1_optuna_trials.csv m1_optuna

Note: only final trial value + params are imported. Per-epoch intermediate
values are not in the CSV and are lost (TPE only uses final values to
propose new trials, so this is fine for the purpose).

Distribution definitions below MUST match what opt_captioning.py uses. Keep
them in sync if the search space ever changes.
"""

import sys
from pathlib import Path

import pandas as pd
import optuna
from optuna.trial import TrialState, create_trial
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
)


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

csv_path = Path(sys.argv[1])
study_name = sys.argv[2]


M1_DISTRIBUTIONS = {
    "lr":              FloatDistribution(1e-5, 5e-3, log=True),
    "dropout":         FloatDistribution(0.05, 0.5),
    "weight_decay":    FloatDistribution(1e-7, 1e-3, log=True),
    "encoder_dropout": FloatDistribution(0.1, 0.3),
    "hidden_dim":      CategoricalDistribution([256, 384, 512, 768, 1024]),
    "num_lstm_layers": CategoricalDistribution([1, 2]),
}

M4_DISTRIBUTIONS = {
    "lr":                 FloatDistribution(1e-5, 5e-3, log=True),
    "dropout":            FloatDistribution(0.05, 0.5),
    "weight_decay":       FloatDistribution(1e-7, 1e-3, log=True),
    "encoder_dropout":    FloatDistribution(0.1, 0.3),
    "d_model":            CategoricalDistribution([256, 384, 512, 768]),
    "nhead":              CategoricalDistribution([4, 8, 16]),
    "num_decoder_layers": CategoricalDistribution([2, 4, 6]),
    "dim_feedforward":    CategoricalDistribution([1024, 2048, 4096]),
}


name_lc = study_name.lower()
if "m1" in name_lc:
    distributions = M1_DISTRIBUTIONS
elif "m4" in name_lc:
    distributions = M4_DISTRIBUTIONS
else:
    raise ValueError(
        f"Cannot infer distributions from study_name {study_name!r}. "
        "Expected 'm1' or 'm4' in the name."
    )


# Storage path matches opt_captioning.py
ROOT = Path(__file__).resolve().parent
storage_dir = ROOT / "optuna_studies"
storage_dir.mkdir(exist_ok=True)
storage = f"sqlite:///{storage_dir / f'{study_name}.db'}"

study = optuna.create_study(
    storage=storage,
    study_name=study_name,
    direction="minimize",
    load_if_exists=True,
)

print(f"Loading {csv_path} into study {study_name!r}")
print(f"Storage: {storage}")
print(f"Existing trials in study before import: {len(study.trials)}")

df = pd.read_csv(csv_path)
print(f"CSV has {len(df)} rows")

n_added = 0
n_skipped = 0

for _, row in df.iterrows():
    state_str = row["state"]

    if state_str == "COMPLETE":
        state = TrialState.COMPLETE
    elif state_str == "PRUNED":
        state = TrialState.PRUNED
    else:
        # FAIL or RUNNING -- skip
        n_skipped += 1
        continue

    params = {}
    for col in df.columns:
        if not col.startswith("params_"):
            continue
        param_name = col[len("params_"):]
        if param_name not in distributions:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        # Categoricals from CSV come back as float for int-typed values
        dist = distributions[param_name]
        if isinstance(dist, CategoricalDistribution):
            sample = dist.choices[0]
            if isinstance(sample, int):
                v = int(v)
        params[param_name] = v

    value = row.get("value")
    if state == TrialState.COMPLETE:
        if value is None or pd.isna(value):
            n_skipped += 1
            continue
        value = float(value)
    else:
        value = float(value) if not pd.isna(value) else None

    try:
        trial = create_trial(
            params=params,
            distributions=distributions,
            value=value,
            state=state,
        )
        study.add_trial(trial)
        n_added += 1
    except Exception as e:
        print(f"  skipping trial {row.get('number', '?')}: {e}")
        n_skipped += 1


print()
print(f"Added: {n_added}")
print(f"Skipped: {n_skipped}")
print(f"Total trials in study now: {len(study.trials)}")
if len(study.trials) > 0 and study.best_trial.value is not None:
    print(f"Best trial value: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
