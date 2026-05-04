### Environment Setup

conda env create -f environment.yml

conda activate image-captioning

After creating the conda environment,
run a one-time NLTK resource download:

	import nltk
	nltk.download("wordnet")
	nltk.download("omw-1.4")


### Data download and preparation:

	1. Flickr8k_Dataset.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
	2. Flickr8k_text.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

Create a raw folder in datasets/flickr8k/, and run:

	python -m data.prepare_flickr8k

Data is already partitioned into training, validation and test by running the script.

### Embeddings download:

download glove.6B.300d.txt to datasets/embeddings/ folder

### m_Config setup to select different models:

Select "Global" vs "Spatial" for CNN encoder

Select "LSTM" vs "Transformer" for decoder

Set attention = True for LSTM with attention.

M1: CNN Global encoder + LSTM decoder

M2: CNN Spatial encoder + LSTM attention decoder

M3: CNN Spatial encoder + Transformer decoder

M4: CNN Global encoder + Transformer decoder

### Training:

To train the model, run:

	python -m training.m_train.py

outputs folder will be created to store checkpoints for each model after training is done.

### Evaluation:

To evaluate metrics for each model, run:

	python -m evaluation.m_eval
	
### Notes:

Because the image data files, embeddings and checkpoints results are too large, they are not uploaded into this repository.

Please download the data and embedings first before initiating the training process for each model.





