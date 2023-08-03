"""
Usage:

python main.py --config my_file.yaml
"""
from utils import read_in_sentences
from create_embeddings_and_clusters import k_means_clustering

path_to_input_file = "data/iza_sentences.txt"
n_clusters = 6
model_name = 'all-MiniLM-L6-v2'



def main():
    # read in sentences
    list_of_sentences = read_in_sentences(path_to_input_file)
    clustered_sentences = k_means_clustering(model_name, list_of_sentences, n_clusters)
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster", i+1, "\t", cluster)
        print("\n")


if __name__=="__main__":
    main()



