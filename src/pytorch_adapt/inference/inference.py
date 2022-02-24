import torch

from ..utils.common_functions import check_domain


# features and logits
def default_fn(x, models, **kwargs):
    features = models["G"](x)
    logits = models["C"](features)
    return {"features": features, "logits": logits}


# adabn features and logits
def adabn_fn(x, domain, models, **kwargs):
    domain = check_domain(domain, keep_len=True)
    features = models["G"](x, domain)
    logits = models["C"](features, domain)
    return {"features": features, "logits": logits}


# adda features and logits
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


# rtn features and logits
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


# mcd features and logits
def mcd_fn(x, models, return_logits_list=False, **kwargs):
    """
    Returns:
        Features and logits, where ```logits = sum(C(features))```.
    """
    features = models["G"](x)
    logits_list = models["C"](features)
    logits = sum(logits_list)
    output = {"features": features, "logits": logits}
    if return_logits_list:
        output["logits_list"] = logits_list
    return output


# mcd features and logits, and the logits from each classifier
def mcd_full_fn(**kwargs):
    return mcd_fn(return_logits_list=True, **kwargs)


# symnets features and logits
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


# discriminator logits
def d_fn(x, models, **kwargs):
    return {"d_logits": models["D"](x)}


# output of a feature combiner
def feature_combiner_fn(x, y, misc, **kwargs):
    return {"features_logits_combined": misc["feature_combiner"](x, y)}


# output of fn and d_fn
def with_d(x, models, fn, layer="features", **kwargs):
    output = fn(x=x, models=models, layer=layer, **kwargs)
    output2 = d_fn(x=output[layer], models=models, **kwargs)
    return {**output, **output2}


# features, logits, and discriminator logits
def default_with_d(**kwargs):
    return with_d(fn=default_fn, **kwargs)


# features, logits, and discriminator logits, where D takes in the output of C
def default_with_d_logits_layer(**kwargs):
    return with_d(fn=default_fn, layer="logits", **kwargs)


# adda features, logits, and discriminator logits
def adda_with_d(**kwargs):
    return with_d(fn=adda_fn, **kwargs)


# output of fn and a feature combiner that takes features and logits or preds as input
def with_feature_combiner(x, models, misc, fn, softmax=True, **kwargs):
    output = fn(x=x, models=models, misc=misc, softmax=softmax, **kwargs)
    logits = output["logits"]
    if softmax:
        logits = torch.softmax(logits, dim=1)
    output2 = feature_combiner_fn(x=output["features"], y=logits, misc=misc, **kwargs)
    return {**output, **output2}


# cdan features, logits, discriminator logits, and feature_combiner output
def cdan_full_fn(**kwargs):
    return with_feature_combiner(fn=default_with_d, **kwargs)


# discriminator bridge logits
def d_bridge_fn(x, models, **kwargs):
    [d_logits, d_bridge] = models["D"](x, return_bridge=True)
    return {"d_logits": d_logits, "d_bridge": d_bridge}


# output of fn and d_bridge_fn. D takes in preds
def with_d_bridge(x, models, fn, **kwargs):
    output = fn(x=x, models=models, **kwargs)
    preds = torch.softmax(output["logits"], dim=1)
    output2 = d_bridge_fn(x=preds, models=models, **kwargs)
    return {**output, **output2}


# gvb features, logits, and g_bridge
def gvb_with_g_bridge(x, models, **kwargs):
    features = models["G"](x)
    [logits, bridge] = models["C"](features, return_bridge=True)
    return {"features": features, "logits": logits, "g_bridge": bridge}


# gvb features, logits, g_bridge, and d_bridge
def gvb_full_fn(**kwargs):
    return with_d_bridge(fn=gvb_with_g_bridge, **kwargs)
