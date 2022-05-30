from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN
from sklearn.cluster import KMeans


def get_knn_func():
    return CustomKNN(LpDistance(normalize_embeddings=False))


def kmeans_func(query, num_clusters):
    return KMeans(n_clusters=num_clusters).fit_predict(query.cpu().numpy())
