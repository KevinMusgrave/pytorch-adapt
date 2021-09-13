# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch

from pytorch_adapt.models import MNISTFeatures


def mnist(model_name):
    url = {
        "C": "https://cornell.box.com/shared/static/j4zrogronmievq1csulrkai7zjm27gcq",
        "G": "https://cornell.box.com/shared/static/tdx0ts24e273j7mf3r2ox7a12xh4fdfy",
    }[model_name]

    if model_name == "C":
        model = Classifier(num_classes=10, in_size=1200, h=256)
    elif model_name == "G":
        model = MNISTFeatures()

    model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))
    return model
