# ==========================================
# M1 Configuration
# CNN Global Encoder + LSTM Decoder (no attention)
# Show and Tell baseline (Vinyals et al., 2015)
# ==========================================


# ==========================================
# Model Hyperparameters
# ==========================================

# LSTM decoder
hidden_dim = 512
num_lstm_layers = 1   # Show and Tell originally uses a single layer
dropout = 0.1
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
embedding_type = "pretrained_frozen"

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
num_epochs = 10

learning_rate = 1e-4
weight_decay = 0.0
grad_clip = 1.0

freeze_backbone = True
pretrained_encoder = True

num_workers = 0


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

debug = True

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
