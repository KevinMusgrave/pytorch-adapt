import torch


def create_output_dict(features, logits):
    return {
        "features": features,
        "logits": logits,
        "preds": torch.softmax(logits, dim=1),
    }


def create_output_dict_preds_as_features(features, logits):
    if not torch.allclose(features, logits):
        raise ValueError(f"features and logits should be equal")
    return {
        "features": features,
        "preds": features,
    }


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
        features, logits = inference(data["imgs"], domain=data["domain"])
    output = output_dict_fn(features, logits)
    if len(output.keys() & data.keys()) > 0:
        raise ValueError("output and data should have no overlap at this point")
    output.update(data)
    return output


def filter_datasets(datasets, validator):
    return {
        k: v for k, v in datasets.items() if k in ["train"] + validator.required_data
    }
