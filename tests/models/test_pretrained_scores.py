import unittest
from collections import defaultdict

import numpy as np
import torch
import tqdm

from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm
from pytorch_adapt.models import (
    mnistC,
    mnistG,
    office31C,
    office31G,
    officehomeC,
    officehomeG,
    pretrained_src_accuracy,
    pretrained_target_accuracy,
)
from pytorch_adapt.validators import AccuracyValidator

from .. import DATASET_FOLDER, TEST_DEVICE


def get_acc_fn(num_classes):
    def get_acc_dict(preds, labels):
        output = {}
        for average in ["macro", "micro"]:
            output[average] = AccuracyValidator(
                torchmetric_kwargs={"average": average, "num_classes": num_classes}
            )(src_val={"preds": preds, "labels": labels})
        return output

    return get_acc_dict


class TestPretrainedScores(unittest.TestCase):
    def helper(
        self, dataset_name, src_domain, G, C, dataset_getter, domains, acc_getter
    ):
        G.eval()
        C.eval()
        accuracies = defaultdict(dict)
        for domain in domains:
            datasets = dataset_getter(
                [domain], [], folder=DATASET_FOLDER, download=True
            )
            datasets.pop("train")
            self.assertTrue(len(datasets) == 2)  # should have src_train and src_val
            dataloaders = DataloaderCreator(num_workers=0, batch_size=64, all_val=True)(
                **datasets
            )
            for k, v in dataloaders.items():
                k = k.split("_")[1]  # remove domain identifier
                print(f"collecting {domain} {k}")
                preds, all_labels = [], []
                for data in tqdm.tqdm(v):
                    imgs, labels = data["src_imgs"], data["src_labels"]
                    preds.append(torch.softmax(C(G(imgs)), dim=1))
                    all_labels.append(labels)
                preds = torch.cat(preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                accuracies[domain][k] = acc_getter(preds, all_labels)

        for domain, x in accuracies.items():
            for split, y in x.items():
                for average, acc in y.items():
                    print(f"checking pretrained acc for {domain} {split} {average}")
                    acc = np.round(acc, 4)
                    if domain == src_domain:
                        true_acc = pretrained_src_accuracy(
                            dataset_name, [domain], split, average
                        )
                    else:
                        true_acc = pretrained_target_accuracy(
                            dataset_name, [src_domain], [domain], split, average
                        )
                    self.assertEqual(acc, true_acc)

    @torch.no_grad()
    def test_mnist(self):
        G = mnistG(pretrained=True, map_location=TEST_DEVICE)
        C = mnistC(pretrained=True, map_location=TEST_DEVICE)
        self.helper(
            "mnist",
            "mnist",
            G,
            C,
            get_mnist_mnistm,
            ["mnist", "mnistm"],
            get_acc_fn(num_classes=10),
        )
