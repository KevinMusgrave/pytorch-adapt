import torch

from ..datasets import (
    CombinedSourceAndTargetDataset,
    DataloaderCreator,
    PseudoLabeledDataset,
    TargetDataset,
)
from ..utils import common_functions as c_f


def get_predicted_labels(adapter, datasets, dataset_name, dataloader_creator):
    dataloader = dataloader_creator(**{dataset_name: datasets[dataset_name]})
    dataloader = dataloader[dataset_name]
    c_f.val_dataloader_checks(dataloader)
    logits = adapter.get_all_outputs(dataloader, dataset_name)[dataset_name]["logits"]
    return torch.argmax(logits, dim=1).cpu().numpy()


def get_pseudo_labeled_dataset(adapter, datasets, dataset_name, dataloader_creator):
    pseudo_labels = get_predicted_labels(
        adapter, datasets, dataset_name, dataloader_creator
    )
    return PseudoLabeledDataset(datasets[dataset_name].dataset, pseudo_labels)


class ReverseValidator:
    def __init__(self):
        self.pseudo_train = None
        self.pseudo_val = None

    def run(
        self,
        forward_adapter,
        reverse_adapter,
        forward_kwargs,
        reverse_kwargs,
        pl_dataloader_creator=None,
    ):
        if "datasets" in reverse_kwargs:
            raise KeyError(
                "'datasets' should not be in reverse_kwargs because the reverse datasets will be pseudo labeled."
            )
        if "validator" not in reverse_kwargs:
            raise KeyError("reverse_kwargs must include 'validator'")

        forward_adapter.run(**forward_kwargs)
        if all(x in forward_kwargs for x in ["validator", "saver"]):
            forward_kwargs["saver"].load_adapter(forward_adapter.adapter, "best")

        datasets = forward_kwargs["datasets"]
        pl_dataloader_creator = c_f.default(
            pl_dataloader_creator, DataloaderCreator, {"all_val": True}
        )

        d = {}
        d["src_train"] = get_pseudo_labeled_dataset(
            forward_adapter, datasets, "target_train", pl_dataloader_creator
        )
        d["src_val"] = get_pseudo_labeled_dataset(
            forward_adapter, datasets, "target_val", pl_dataloader_creator
        )
        d["target_train"] = TargetDataset(datasets["src_train"].dataset)
        d["target_val"] = TargetDataset(datasets["src_val"].dataset)
        d["train"] = CombinedSourceAndTargetDataset(d["src_train"], d["target_train"])

        self.pseudo_train = d["src_train"]
        self.pseudo_val = d["src_val"]

        reverse_kwargs["datasets"] = d
        return reverse_adapter.run(**reverse_kwargs)
