# ==========================================
# M1 Configuration
# CNN Global Encoder + LSTM Decoder (no attention)
# Show and Tell baseline (Vinyals et al., 2015)
# ==========================================


# ==========================================
# Model Hyperparameters
# ==========================================

# LSTM decoder
# Best HPs from optuna (m1_optuna trial 1, val_loss=3.088 at data_pct=0.25, 10 epochs).
# Show and Tell originally uses a single layer; the search picked 2 layers / 1024 hidden.
hidden_dim = 1024
num_lstm_layers = 2
dropout = 0.31
encoder_dropout = 0.21
max_len = 50

# Encoder-decoder interface
encoder_dim = 512


# ==========================================
# Word Embedding Strategy
# ==========================================

# Options:
#   "random"
#   "pretrained_frozen"
#   "pretrained_finetune"
# Using "random" for the first end-to-end validation run since we don't have
# GloVe downloaded yet, and this matches what the optuna study used.
embedding_type = "random"

# GloVe 6B 300d
embedding_dim = 300
embedding_path = "datasets/embeddings/glove.6B.300d.txt"


# ==========================================
# Experiment Name
# ==========================================

if embedding_type == "random":
    experiment_name = "m1_random"

elif embedding_type == "pretrained_frozen":
    experiment_name = "m1_glove_frozen"

elif embedding_type == "pretrained_finetune":
    experiment_name = "m1_glove_finetune"

else:
    raise ValueError(f"Unsupported embedding_type: {embedding_type}")


# ==========================================
# Vocabulary placeholders
# filled dynamically from dataset
# ==========================================

vocab_size = None
pad_idx = None
start_idx = None
end_idx = None
unk_idx = None


# ==========================================
# Dataset
# ==========================================

dataset_name = "flickr8k"
image_folder = "Flicker8k_Dataset"

freq_threshold = 5
image_size = 224


# ==========================================
# Training Hyperparameters
# ==========================================

batch_size = 32
num_epochs = 20

# Optuna best lr was 4.17e-4 at data_pct=0.25, 10 epochs (training tokens = 75k).
# We're scaling to data_pct=1.0, 20 epochs (training tokens = 600k), so kappa=8.
# Per the Complete(d)P paper (arxiv 2512.22382), divide LR by sqrt(kappa) ~= 2.83.
# 4.17e-4 / 2.83 ~= 1.5e-4.
learning_rate = 1.5e-4
weight_decay = 4.4e-6
grad_clip = 1.0

freeze_backbone = True
pretrained_encoder = True

num_workers = 2


# ==========================================
# Output Structure
# ==========================================

checkpoint_dir = "outputs/checkpoints"

best_checkpoint_name = f"{experiment_name}_best.pt"
last_checkpoint_name = f"{experiment_name}_last.pt"

log_dir = f"outputs/logs/M1/{experiment_name}"
train_log_name = "train_log.txt"

prediction_dir = f"outputs/predictions/M1/{experiment_name}"

val_prediction_name = "val_predictions.json"
test_prediction_name = "test_predictions.json"
metrics_result_name = "metrics_results.json"


# ==========================================
# Debug / Smoke Test Mode
# ==========================================

debug = False

debug_batch_size = 8
debug_num_epochs = 1
debug_pretrained_encoder = False


# ==========================================
# Optional Training Utilities
# ==========================================

use_scheduler = False
scheduler_step_size = 5
scheduler_gamma = 0.5

early_stopping_patience = 5
