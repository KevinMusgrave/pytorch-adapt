import unittest

from pytorch_adapt.models import (
    mnistC,
    mnistG,
    office31C,
    office31G,
    officehomeC,
    officehomeG,
)


class TestPretrainedDownload(unittest.TestCase):
    def test_pretrained_download(self):
        for x in [mnistC, mnistG, office31G, officehomeG]:
            x(pretrained=True)

        for domain in ["amazon", "dslr", "webcam"]:
            office31C(domain=domain, pretrained=True)

        for domain in ["art", "clipart", "product", "real"]:
            officehomeC(domain=domain, pretrained=True)
