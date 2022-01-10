from .classifier import Classifier
from .mnist import MNISTFeatures
from .utils import download_weights


def mnistG(pretrained=False, progress=True):
    model = MNISTFeatures()
    url = "https://cornell.box.com/shared/static/tdx0ts24e273j7mf3r2ox7a12xh4fdfy"
    h = "68ee79452f1d5301be2329dfa542ac6fa18de99e09d6540838606d9d700b09c8"
    filename = f"mnistG-{h[:8]}.pth"
    return download_weights(model, url, pretrained, progress, filename)


def mnistC(num_classes=10, in_size=1200, h=256, pretrained=False, progress=True):
    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    url = "https://cornell.box.com/shared/static/j4zrogronmievq1csulrkai7zjm27gcq"
    h = "ac7b5a13df2ef3522b6550a147eb44dde8ff4fead3ddedc540d9fe63c9d597c1"
    filename = f"mnistC-{h[:8]}.pth"
    return download_weights(model, url, pretrained, progress, filename)


def resnet50(pretrained=False, progress=True):
    import timm

    model = timm.create_model("resnet50", pretrained=False, num_classes=0)

    # G was frozen during finetuning
    # So the model for all 3 domains is the same
    url = "https://cornell.box.com/shared/static/1oxb5xk5dq3od1d3gprigznxmqb3wgr1"
    h = "a567ecd6ea5addf29ccfa8bf706be78a35adff7cd5cf5a3a99d89b19807454ae"
    filename = f"resnet50MusgraveUDA-{h[:8]}.pth"
    return download_weights(model, url, pretrained, progress, filename)


def office31G(*args, **kwargs):
    return resnet50(*args, **kwargs)


def office31C(
    domain, num_classes=31, in_size=2048, h=256, pretrained=False, progress=True
):
    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    url = {
        "amazon": "https://cornell.box.com/shared/static/6h165jqlxcpo16jbs3a7vpvslb6u9vaq",
        "dslr": "https://cornell.box.com/shared/static/t97sedzf4wrto3yfvr8hxivyblqkljiq",
        "webcam": "https://cornell.box.com/shared/static/zuv7be39v8bijwggrvfzlyw1h0pfwrb4",
    }[domain]
    h = {
        "amazon": "6e2fb6f392538172515c2c673a8b3ead7aad8b88b44aad6468c7e9b11761b667",
        "dslr": "fc0acd7a71eb5f12d4af619e5c63bcc42e5a23441bbd105fe0f7a37c26f37d80",
        "webcam": "b2bb55978380fa9ca6452cba30e0ac2a19b7166d8348bcc1554fdabd185e4cdd",
    }[domain]
    filename = f"office31C{domain}-{h[:8]}.pth"
    return download_weights(model, url, pretrained, progress, filename)


def officehomeG(*args, **kwargs):
    return resnet50(*args, **kwargs)


def officehomeC(
    domain, num_classes=65, in_size=2048, h=256, pretrained=False, progress=True
):
    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    url = {
        "art": "https://cornell.box.com/shared/static/wxg7v32e2m0jcmq53amhdipty9veb2xx",
        "clipart": "https://cornell.box.com/shared/static/4dhwhj6fkzg9lfgu0mfskt2kby8mznez",
        "product": "https://cornell.box.com/shared/static/r6f3ltgve5g2lrcdtoykj84rlyzqs6ga",
        "real": "https://cornell.box.com/shared/static/1lf1foq65m77pdpc50isdgsc8k71ei29",
    }[domain]
    h = {
        "art": "8db546ff250d2b54899f92e482e80a68411cbe525134d429987b57d3b0571e4b",
        "clipart": "8e145cc6d2df3ff428aeafa43066bbde97d56e9f844b34408bdca74125e62590",
        "product": "472ff36fdf13ec6c1fa1236d1d0800e2a5cf2e3d366b6b63ff5807dff6a761d8",
        "real": "f0c8d6e941d4f488ff2438eb5cccdc59e78f35961e48f03d2186752e5878c697",
    }[domain]
    filename = f"officehomeC{domain}-{h[:8]}.pth"
    return download_weights(model, url, pretrained, progress, filename)
