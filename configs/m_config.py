# ==========================================
# Configuration:
#
# CNN Spatial Encoder + Transformer Decoder
# CNN Global Encoder + Transformer
# ==========================================


# ==========================================
# Model Hyperparameters
# ==========================================

# Encoder-decoder selection
encoder_type = "Spatial" # Global vs Spatial
decoder_type = "LSTM" # LSTM vs Transformer

model_type = encoder_type + "_" + decoder_type

LSTM_Attention = True # Attention for LSTM decoder

# LSTM decoder
hidden_dim = 512
num_lstm_layers = 2
attention_dim = 512


# Transformer decoder
d_model = 512
nhead = 8
num_decoder_layers = 4
dim_feedforward = 2048
dropout = 0.1
max_len = 50

# Encoder-decoder interface
encoder_dim = 512
encoder_spatial_size = 49 # ???? not sure
use_positional_encoding = True


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
    experiment_name = model_type + "_random"

elif embedding_type == "pretrained_frozen":
    experiment_name = model_type + "_glove_frozen"

elif embedding_type == "pretrained_finetune":
    experiment_name = model_type + "_glove_finetune"

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

log_dir = f"outputs/logs/{experiment_name}"
train_log_name = "train_log.txt"

prediction_dir = f"outputs/predictions/{experiment_name}"

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
