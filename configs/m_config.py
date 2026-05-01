# ==========================================
# Configuration:
# Unified Image Captioning Framework
# ==========================================


# ==========================================
# Encoder / Decoder Selection
# ==========================================

# Options:
#   encoder_type: "Global" | "Spatial"
#   decoder_type: "LSTM" | "Transformer"
encoder_type = "Spatial"
decoder_type = "Transformer"

# Only used when decoder_type == "LSTM"
LSTM_Attention = True


# ==========================================
# Model Naming
# ==========================================

if decoder_type == "LSTM":
    if LSTM_Attention:
        decoder_name = "LSTM_Attention"
    else:
        decoder_name = "LSTM"

elif decoder_type == "Transformer":
    decoder_name = "Transformer"

else:
    raise ValueError(f"Unsupported decoder_type: {decoder_type}")

model_type = encoder_type + "_" + decoder_name


# ==========================================
# LSTM Hyperparameters
# ==========================================

hidden_dim = 512
num_lstm_layers = 2
attention_dim = 512


# ==========================================
# Transformer Hyperparameters
# ==========================================

d_model = 512
nhead = 8
num_decoder_layers = 4
dim_feedforward = 2048
dropout = 0.1
max_len = 50


# ==========================================
# Encoder Interface
# ==========================================

encoder_dim = 512
encoder_spatial_size = 49
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
# Vocabulary Placeholders
# Filled dynamically from dataset
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
weight_decay = 1e-4
grad_clip = 1.0

freeze_backbone = True
pretrained_encoder = True

num_workers = 0


# ==========================================
# Optimizer
# ==========================================

# Options:
#   "adam"
#   "adamw"
#   "sgd"
optimizer_type = "adamw"

# Only used when optimizer_type == "sgd"
momentum = 0.9


# ==========================================
# Optional Training Utilities
# ==========================================

use_scheduler = False
scheduler_step_size = 5
scheduler_gamma = 0.5

early_stopping_patience = 5


# ==========================================
# Output Structure
# ==========================================

plot_dir = "outputs"

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
debug_num_epochs = 2
debug_pretrained_encoder = True


# ==========================================
# Decoding / Evaluation Strategy
# ==========================================

# Options:
#   "greedy"
#   "beam"
decode_strategy = "beam"

# Beam search parameters
beam_size = 3
length_penalty = 0.7