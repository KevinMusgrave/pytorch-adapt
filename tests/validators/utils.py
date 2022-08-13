import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as pml_cf
from pytorch_metric_learning.utils.inference import CustomKNN
from sklearn.cluster import KMeans

from pytorch_adapt.adapters import Classifier
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import (
    CombinedSourceAndTargetDataset,
    SourceDataset,
    TargetDataset,
)
from pytorch_adapt.frameworks.ignite import (
    Ignite,
    IgniteMultiLabelClassification,
    IgnitePredsAsFeatures,
)
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import ScoreHistory

from .. import TEST_DEVICE


def get_knn_func():
    return CustomKNN(LpDistance(normalize_embeddings=False))


def kmeans_func(query, num_clusters):
    return KMeans(n_clusters=num_clusters).fit_predict(query.cpu().numpy())


def test_with_ignite_framework(
    validator,
    assertion_fn,
    num_classes=10,
    multilabel=False,
    adapter_cls=None,
    ignite_cls_list=None,
):
    ignite_cls_list = c_f.default(
        ignite_cls_list, [Ignite, IgnitePredsAsFeatures, IgniteMultiLabelClassification]
    )
    adapter_cls = c_f.default(adapter_cls, Classifier)

    for wrapper_type in ignite_cls_list:
        c_f.LOGGER.info(f"Testing with wrapper class {wrapper_type}")
        dataset_size = 9999

        datasets = {}
        features = {}
        labels = {}
        for split in ["src_train", "src_val", "target_train", "target_val"]:
            features[split] = torch.randn(dataset_size, 128)
            if multilabel:
                labels[split] = torch.randint(0, 2, size=(dataset_size, num_classes))
            else:
                labels[split] = torch.randint(0, num_classes, size=(dataset_size,))
            datasets[split] = pml_cf.EmbeddingDataset(features[split], labels[split])
            if split.startswith("src"):
                datasets[split] = SourceDataset(datasets[split])
            else:
                datasets[split] = TargetDataset(datasets[split])

        datasets["train"] = CombinedSourceAndTargetDataset(
            datasets["src_train"], datasets["target_train"]
        )

        G = torch.nn.Linear(128, 128).to(TEST_DEVICE)
        C = torch.nn.Linear(128, num_classes).to(TEST_DEVICE)

        if wrapper_type is IgnitePredsAsFeatures:
            preds_fn = torch.nn.Sigmoid() if multilabel else torch.nn.Softmax(dim=1)
            G = torch.nn.Sequential(G, C, preds_fn)
            C = torch.nn.Identity()

        models = Models({"G": G, "C": C})
        optimizers = Optimizers((torch.optim.Adam, {"lr": 0}))
        adapter = wrapper_type(
            adapter_cls(models=models, optimizers=optimizers),
            validator=ScoreHistory(validator),
            device=TEST_DEVICE,
        )
        score, _ = adapter.run(
            datasets,
            epoch_length=1,
        )

        logits = {}
        for k, v in features.items():
            with torch.no_grad():
                if wrapper_type is IgnitePredsAsFeatures:
                    logits[k] = torch.nn.Sequential(*G[:2])(v.to(TEST_DEVICE))
                else:
                    logits[k] = C(G(v.to(TEST_DEVICE)))

        assertion_fn(logits, labels, score)
