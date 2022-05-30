from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN


def get_knn_func():
    return CustomKNN(LpDistance(normalize_embeddings=False))
