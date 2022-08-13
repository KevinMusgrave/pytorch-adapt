import torch

from ..utils import common_functions as c_f


def create_output_dict(f_dict):
    return {**f_dict, "preds": torch.softmax(f_dict["logits"], dim=1)}


def create_output_dict_multilabel_classification(f_dict):
    return {**f_dict, "preds": torch.sigmoid(f_dict["logits"])}


def create_output_dict_preds_as_features(f_dict):
    [features, logits] = c_f.extract(f_dict, ["features", "logits"])
    if not torch.allclose(features, logits):
        raise ValueError("features and logits should be equal")
    return {"features": features, "preds": features}


def extract_data(batch):
    data = {"_".join(k.split("_")[1:]): v for k, v in batch.items()}
    if len(data) != len(batch):
        raise KeyError(
            f"Batch should have only one domain, but it has keys: {batch.keys()}"
        )
    return data


def collector_step(inference, batch, output_dict_fn):
    data = extract_data(batch)
    with torch.no_grad():
        f_dict = inference(data["imgs"], domain=data["domain"])
    data.pop("imgs")  # we don't want to collect imgs
    f_dict = output_dict_fn(f_dict)
    if len(f_dict.keys() & data.keys()) > 0:
        raise ValueError("f_dict and data should have no overlap at this point")
    f_dict.update(data)
    return f_dict


def filter_datasets(datasets, validator):
    return {
        k: v for k, v in datasets.items() if k in ["train"] + validator.required_data
    }
