# ==========================================
# Configuration:
# Unified Image Captioning Framework
# ==========================================


# ==========================================
# Encoder / Decoder Selection
# ==========================================

encoder_type = "Spatial"      # "Global" | "Spatial"
decoder_type = "Transformer"  # "LSTM" | "Transformer"

# LSTM specific
LSTM_Attention = True  # True only matters when decoder_type == "LSTM"


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
# Transformer Hyperparameters
# ==========================================

d_model = 512
nhead = 8
num_decoder_layers = 4
dim_feedforward = 2048
dropout = 0.1
max_len = 50


# ==========================================
# LSTM Hyperparameters
# ==========================================

hidden_dim = 512
num_lstm_layers = 2
attention_dim = 512


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

# GloVe
embedding_dim = 300
embedding_path = "datasets/embeddings/glove.6B.300d.txt"


# ==========================================
# Experiment Name (自动生成)
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

plot_dir = "outputs"

checkpoint_dir = "outputs/checkpoints"

best_checkpoint_name = f"{experiment_name}_best.pt"
last_checkpoint_name = f"{experiment_name}_last.pt"

log_dir = f"outputs/logs/{experiment_name}"

prediction_dir = f"outputs/predictions/{experiment_name}"

val_prediction_name = "val_predictions.json"
test_prediction_name = "test_predictions.json"
metrics_result_name = "metrics_results.json"


# ==========================================
# Debug Mode
# ==========================================

debug = False

debug_batch_size = 32
debug_num_epochs = 2
debug_pretrained_encoder = True


# ==========================================
# Decoding / Evaluation Strategy
# ==========================================

# "greedy" | "beam"
decode_strategy = "beam"

beam_size = 3
length_penalty = 0.7