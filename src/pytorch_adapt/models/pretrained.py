from .classifier import Classifier
from .mnist import MNISTFeatures
from .utils import download_weights


def mnistG(pretrained=False, progress=True, **kwargs):
    """
    Returns:
        An [```MNISTFeatures```][pytorch_adapt.models.MNISTFeatures] model
        trained on the MNIST dataset, if ```pretrained == True```.
    """
    model = MNISTFeatures()
    url = "https://cornell.box.com/shared/static/tdx0ts24e273j7mf3r2ox7a12xh4fdfy"
    h = "68ee79452f1d5301be2329dfa542ac6fa18de99e09d6540838606d9d700b09c8"
    file_name = f"mnistG-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def mnistC(
    num_classes=10, in_size=1200, h=256, pretrained=False, progress=True, **kwargs
):
    """
    Returns:
        A [```Classifier```][pytorch_adapt.models.Classifier] model
        trained on the MNIST dataset, if ```pretrained == True```.
    """
    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    url = "https://cornell.box.com/shared/static/j4zrogronmievq1csulrkai7zjm27gcq"
    h = "ac7b5a13df2ef3522b6550a147eb44dde8ff4fead3ddedc540d9fe63c9d597c1"
    file_name = f"mnistC-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def resnet50(pretrained=False, progress=True, **kwargs):
    import timm

    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    url = "https://cornell.box.com/shared/static/1oxb5xk5dq3od1d3gprigznxmqb3wgr1"
    h = "a567ecd6ea5addf29ccfa8bf706be78a35adff7cd5cf5a3a99d89b19807454ae"
    file_name = f"resnet50MusgraveUDA-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def office31G(*args, **kwargs):
    """
    Returns:
        A ResNet50 model trained on ImageNet, if ```pretrained == True```.
    """
    # G was frozen during finetuning
    return resnet50(*args, **kwargs)


def office31C(
    domain=None,
    num_classes=31,
    in_size=2048,
    h=256,
    pretrained=False,
    progress=True,
    **kwargs,
):
    """
    Returns:
        A [```Classifier```][pytorch_adapt.models.Classifier] model
        trained on the specified ```domain``` of the [Office31][pytorch_adapt.datasets.Office31]
        dataset, if ```pretrained == True```. For example

        ```python
        model = office31(domain="amazon", pretrained=True)
        ```
    """
    if (pretrained and not domain) or (not pretrained and domain):
        raise ValueError("if pretrained, domain must be specified, and vice versa")

    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    if not pretrained:
        return model
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
    file_name = f"office31C{domain}-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def officehomeG(*args, **kwargs):
    """
    Returns:
        A ResNet50 model trained on ImageNet, if ```pretrained == True```.
    """
    # G was frozen during finetuning
    return resnet50(*args, **kwargs)


def officehomeC(
    domain=None,
    num_classes=65,
    in_size=2048,
    h=256,
    pretrained=False,
    progress=True,
    **kwargs,
):
    """
    Returns:
        A [```Classifier```][pytorch_adapt.models.Classifier] model
        trained on the specified ```domain``` of the [OfficeHome][pytorch_adapt.datasets.OfficeHome]
        dataset, if ```pretrained == True```. For example

        ```python
        model = officehomeC(domain="art", pretrained=True)
        ```
    """
    if (pretrained and not domain) or (not pretrained and domain):
        raise ValueError("if pretrained, domain must be specified, and vice versa")

    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    if not pretrained:
        return model
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
    file_name = f"officehomeC{domain}-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def domainnet126G(domain=None, pretrained=False, progress=True, **kwargs):
    """
    Returns:
        A ResNet50 model trained on ImageNet, if `pretrained == True`.
        If `pretrained == True` and `domain` is specified, then it returns a ResNet50 model trained on that domain.
        For example:
        ```python
        model = domainnet126G(domain="sketch", pretrained=True)
        ```
    """
    import timm

    if not domain:
        return timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
    elif domain and pretrained:
        model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    else:
        raise ValueError("if domain is specified, then pretrained must be True")
    url = {
        "clipart": "https://cornell.box.com/shared/static/p89emqp3sggj8t1ir8ojjgnmgu3i1h6e",
        "painting": "https://cornell.box.com/shared/static/dytk5919v2rwmzgbz8y34hwrtwdh7zbw",
        "real": "https://cornell.box.com/shared/static/cnwd1l38zamp2rwfykrvbw05ds9bq5vp",
        "sketch": "https://cornell.box.com/shared/static/2ifhf99o9oi45c7f2xfspvze9fnbxiwr",
    }[domain]
    h = {
        "clipart": "1cc22355555ae8cafc812584fed828030809ac7eab105c63d220503e4b28e208",
        "painting": "cd4c23e3e8c66900b7cb45d22507f31d43e5da96e9a3e491463f2da72ee15aec",
        "real": "683ac0757a222edab6bad922c7658bc0b1be983699ab38a181f78b150c778315",
        "sketch": "b8c871dc736440aa2e143b75bc540d35056e13012b0de0afb785cba95ed797f7",
    }[domain]
    file_name = f"domainnet126G{domain}-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def domainnet126C(
    domain=None,
    num_classes=126,
    in_size=2048,
    h=256,
    pretrained=False,
    progress=True,
    **kwargs,
):
    """
    Returns:
        A [```Classifier```][pytorch_adapt.models.Classifier] model
        trained on the specified ```domain``` of the [DomainNet126][pytorch_adapt.datasets.DomainNet126]
        dataset, if ```pretrained == True```. For example

        ```python
        model = domainnet126C(domain="sketch", pretrained=True)
        ```
    """
    if (pretrained and not domain) or (not pretrained and domain):
        raise ValueError("if pretrained, domain must be specified, and vice versa")

    model = Classifier(num_classes=num_classes, in_size=in_size, h=h)
    if not pretrained:
        return model
    url = {
        "clipart": "https://cornell.box.com/shared/static/w2bkoubtzusbpy5xrjgtm1g2fky5yrvk",
        "painting": "https://cornell.box.com/shared/static/2hsaja8l7tilnmvil019zz1cfjuqzj99",
        "real": "https://cornell.box.com/shared/static/xjftd2p0tn6m73d5ddgzbpfp5k13psu5",
        "sketch": "https://cornell.box.com/shared/static/sjrqw3c8h4eop7od2998vvn9bor66w12",
    }[domain]
    h = {
        "clipart": "3728c17634360447ede69520df0c02d9037c2d70f4e9eedeb8aba8242f3fb15b",
        "painting": "11af52a20808ae106168493f61c84f2a695895162d45eb0718333eb1c2ac737c",
        "real": "fc9ea8491b486f5c7302baa5db594f372cdb488f279c9f82a42c752c153b55e8",
        "sketch": "313876e52463eca7d3c7c8be125e86374f976f2afca6bd3aaaa75b1547f84bcf",
    }[domain]
    file_name = f"domainnet126C{domain}-{h[:8]}.pth"
    return download_weights(
        model, url, pretrained, progress=progress, file_name=file_name, **kwargs
    )


def voc_multilabelG(*args, **kwargs):
    """
    Returns:
        A ResNet50 model trained on ImageNet, if ```pretrained == True```.
    """
    return resnet50(*args, **kwargs)


def voc_multilabelC(
    domain=None,
    num_classes=20,
    in_size=2048,
    h=256,
    pretrained=False,
    progress=True,
    **kwargs,
):
    if pretrained:
        raise ValueError("pretrained=True not yet supported")

    return Classifier(num_classes=num_classes, in_size=in_size, h=h)
