import torch


def get_silhouette_score(feats, labels):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")
    scores = []
    for L in unique_labels:
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        if num_elements > 1:
            intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                num_elements - 1
            )  # minus 1 to exclude self distance
            dists_to_other_clusters = []
            for otherL in unique_labels:
                if otherL != L:
                    other_cluster = feats[labels == otherL]
                    inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                    mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / (
                        len(other_cluster)
                    )
                    dists_to_other_clusters.append(mean_inter_dists)
            dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
            min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
            curr_scores = (min_dists - mean_intra_dists) / (
                torch.maximum(min_dists, mean_intra_dists)
            )
        else:
            curr_scores = torch.tensor([0], device=device, dtype=dtype)

        scores.append(curr_scores)

    scores = torch.cat(scores, dim=0)
    if len(scores) != num_samples:
        raise ValueError(
            f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
        )
    return torch.mean(scores).item()


class SilhouetteScore(torch.nn.Module):
    """
    A PyTorch implementation of the
    [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
    """

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """ """
        return get_silhouette_score(features, labels)
