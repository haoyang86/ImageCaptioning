conda env create -f environment.yml
conda activate image-captioning


After creating the conda environment,
run a one-time NLTK resource download:
	import nltk
	nltk.download("wordnet")
	nltk.download("omw-1.4")