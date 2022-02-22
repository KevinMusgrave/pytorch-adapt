from ..utils.common_functions import check_domain


def gc_fn(x, models, **kwargs):
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
    f_dict = gc_fn(x=x, models=models)
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


def gcd_fn(x, models, **kwargs):
    output = gc_fn(x, models, **kwargs)
    output.update(d_fn(output["features"], models, **kwargs))
    return output
