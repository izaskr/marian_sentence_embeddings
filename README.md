# Cluster sentences based on their sentence embeddings.

### Installing 
First, create a conda environment,``conda create -n sentence_emb_env python=3.9``

Activate it with ``conda activate sentence_emb_env``

Install the requirements using pip/conda as follows:
- ``conda install pytorch torchvision torchaudio cpuonly -c pytorch``
- ``pip install transformers``
- ``pip install -U sentence-transformers``

## Data
Put your input files into data/ directory. Format: one sentence per line (see example iza_sentences.txt)

## Running the clustering
Currently supported algorithms: K-means clustering. 

In main.py, change the first two lines to point to your file and to create the requested number of clusters.
Run the script while having your environment activated, ``python main.py``