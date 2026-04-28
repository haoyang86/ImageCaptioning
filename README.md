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

Create a raw folder in flickr8k, and run prepare_flickr8k.py

Data is already splitted into training, validation and test by running the script.


### m_Config:

Select "Global" vs "Spatial" for CNN encoder

Select "LSTM" vs "Transformer" for decoder

Set attention = True for LSTM with attention.
