import torch

from ..utils.common_functions import check_domain


def default_fn(x, models, **kwargs):
    features = models["G"](x)
    logits = models["C"](features)
    return {"features": features, "logits": logits}


def adabn_fn(x, domain, models, **kwargs):
    domain = check_domain(domain, keep_len=True)
    features = models["G"](x, domain)
    logits = models["C"](features, domain)
    return {"features": features, "logits": logits}


def adda_fn(x, domain, models, **kwargs):
    """
    Arguments:
        x: The input to the model
        domain: If 0, then ```features = G(x)```
            Otherwise ```features = T(x)```.
    Returns:
        Features and logits
    """
    domain = check_domain(domain)
    fe = "G" if domain == 0 else "T"
    features = models[fe](x)
    logits = models["C"](features)
    return {"features": features, "logits": logits}


def rtn_fn(x, domain, models, **kwargs):
    """
    Arguments:
        x: The input to the model
        domain: If 0, ```logits = residual_model(C(G(x)))```.
            Otherwise, ```logits = C(G(x))```.
    Returns:
        Features and logits
    """
    domain = check_domain(domain)
    f_dict = default_fn(x=x, models=models)
    logits = f_dict["logits"]
    if domain == 0:
        logits = models["residual_model"](logits)
    return {**f_dict, "logits": logits}


def mcd_fn(x, models, **kwargs):
    """
    Returns:
        Features and logits, where ```logits = sum(C(features))```.
    """
    features = models["G"](x)
    logits_list = models["C"](features)
    logits = sum(logits_list)
    return {"features": features, "logits": logits}


def symnets_fn(x, domain, models, **kwargs):
    """
    Arguments:
        x: The input to the model
        domain: 0 for the source domain, 1 for the target domain.
    Returns:
        Features and logits, where ```logits = C(features)[domain]```.
    """
    domain = check_domain(domain)
    features = models["G"](x)
    logits = models["C"](features)[domain]
    return {"features": features, "logits": logits}


def d_fn(x, models, **kwargs):
    return {"d_logits": models["D"](x)}


def feature_combiner_fn(x, y, misc, **kwargs):
    return {"features_logits_combined": misc["feature_combiner"](x, y)}


def with_d(x, models, fn, **kwargs):
    output = fn(x=x, models=models, **kwargs)
    output2 = d_fn(x=output["features"], models=models, **kwargs)
    return {**output, **output2}


def with_feature_combiner(x, models, misc, fn, softmax=True, **kwargs):
    output = fn(x=x, models=models, misc=misc, softmax=softmax, **kwargs)
    logits = output["logits"]
    if softmax:
        logits = torch.softmax(logits, dim=1)
    output2 = feature_combiner_fn(x=output["features"], y=logits, misc=misc, **kwargs)
    return {**output, **output2}


def default_with_d(**kwargs):
    return with_d(fn=default_fn, **kwargs)
