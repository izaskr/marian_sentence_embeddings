import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans


def load_language_model(name='all-MiniLM-L6-v2'):
    """
    :param name: load the encoder of sentences from the sentence_transformers library
    :return:
    """
    model = SentenceTransformer(name)
    return model


def k_means_get_clusters(encoded_sentences, num_clusters=4):
    """
    :param num_clusters: integer, number of clusters
    :param list_of_encoded_sentences: a pytorch tensor of encoded sentences
    :return: clusters
    """
    clustering_kmeans_model = KMeans(n_clusters=num_clusters)
    clustering_kmeans_model.fit(encoded_sentences)
    cluster_assignment = clustering_kmeans_model.labels_
    return cluster_assignment


def k_means_clustering(model_name, sentences, num_clusters):
    """
    :param model_name: str, model name
    :param sentences: list of sentences (str)
    :param num_clusters: int, number of requested clusters
    :return:
    """
    model = load_language_model(model_name)
    encoded_sentences = model.encode(sentences)
    cluster_assignment = k_means_get_clusters(encoded_sentences, num_clusters)
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])
    return clustered_sentences


