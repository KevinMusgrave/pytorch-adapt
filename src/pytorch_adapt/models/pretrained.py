from .classifier import Classifier
from .mnist import MNISTFeatures
from .utils import download_weights


def mnistG(pretrained=False, progress=True, **kwargs):
    model = MNISTFeatures(**kwargs)
    url = "https://cornell.box.com/shared/static/tdx0ts24e273j7mf3r2ox7a12xh4fdfy"
    return download_weights(model, url, pretrained, progress)


def mnistC(num_classes=10, in_size=1200, h=256, pretrained=False, progress=True):
    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    url = "https://cornell.box.com/shared/static/j4zrogronmievq1csulrkai7zjm27gcq"
    return download_weights(model, url, pretrained, progress)
