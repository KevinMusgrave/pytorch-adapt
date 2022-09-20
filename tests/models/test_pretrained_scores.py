import argparse
import pprint
import sys
import unittest
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from pytorch_adapt.datasets import (
    DataloaderCreator,
    get_mnist_mnistm,
    get_office31,
    get_officehome,
)
from pytorch_adapt.datasets.getters import get_domainnet126
from pytorch_adapt.datasets.utils import num_classes
from pytorch_adapt.models import (
    domainnet126C,
    domainnet126G,
    mnistC,
    mnistG,
    office31C,
    office31G,
    officehomeC,
    officehomeG,
    pretrained_src_accuracy,
    pretrained_target_accuracy,
)
from pytorch_adapt.utils.common_functions import batch_to_device
from pytorch_adapt.validators import AccuracyValidator

from .. import DATASET_FOLDER, RUN_PRETRAINED_SCORES_TESTS, TEST_DEVICE

skip_reason = "RUN_PRETRAINED_SCORES_TESTS is False"


def get_acc_dict(dataset_name, preds, labels):
    output = {}
    for average in ["macro", "micro"]:
        output[average] = AccuracyValidator(
            torchmetric_kwargs={
                "average": average,
                "num_classes": num_classes(dataset_name),
            }
        )(src_val={"preds": preds, "labels": labels})
    return output


# just make sure no exceptions are raised
def check_if_keys_exist(dataset_name, src_domain, domains):
    for domain in domains:
        for split in ["train", "val"]:
            for average in ["micro", "macro"]:
                if domain == src_domain:
                    pretrained_src_accuracy(dataset_name, [src_domain], split, average)
                else:
                    pretrained_target_accuracy(
                        dataset_name, [src_domain], [domain], split, average
                    )


class TestPretrainedScores(unittest.TestCase):
    dataset_folder = DATASET_FOLDER
    download = True
    print_output_instead_of_asserting = False

    def helper(
        self, dataset_name, src_domain, G, C, dataset_getter, domains, acc_getter
    ):
        G.eval()
        C.eval()
        accuracies = defaultdict(dict)
        for domain in domains:
            datasets = dataset_getter(
                [domain], [], folder=self.dataset_folder, download=self.download
            )
            datasets.pop("train")
            self.assertTrue(len(datasets) == 2)  # should have src_train and src_val
            dataloaders = DataloaderCreator(num_workers=2, batch_size=32, all_val=True)(
                **datasets
            )
            for k, v in dataloaders.items():
                k = k.split("_")[1]  # remove domain identifier
                print(f"collecting {domain} {k}")
                preds, all_labels = [], []
                for data in tqdm(v):
                    data = batch_to_device(data, TEST_DEVICE)
                    imgs, labels = data["src_imgs"], data["src_labels"]
                    preds.append(torch.softmax(C(G(imgs)), dim=1))
                    all_labels.append(labels)
                preds = torch.cat(preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                accuracies[domain][k] = acc_getter(dataset_name, preds, all_labels)

        pprint.pprint(accuracies)

        for domain, x in accuracies.items():
            for split, y in x.items():
                for average, acc in y.items():
                    print(
                        f"checking pretrained acc for {src_domain} {domain} {split} {average}"
                    )
                    acc = np.round(acc, 4)
                    if domain == src_domain:
                        true_acc = pretrained_src_accuracy(
                            dataset_name, [domain], split, average
                        )
                    else:
                        true_acc = pretrained_target_accuracy(
                            dataset_name, [src_domain], [domain], split, average
                        )
                    if self.print_output_instead_of_asserting:
                        print(acc, true_acc)
                    else:
                        self.assertEqual(acc, true_acc)

    @torch.no_grad()
    @unittest.skipIf(not RUN_PRETRAINED_SCORES_TESTS, skip_reason)
    def test_mnist(self):
        G = mnistG(pretrained=True, map_location=TEST_DEVICE)
        C = mnistC(pretrained=True, map_location=TEST_DEVICE)
        dataset_name = "mnist"
        src_domain = "mnist"
        domains = ["mnist", "mnistm"]
        check_if_keys_exist(dataset_name, src_domain, domains)
        self.helper(
            dataset_name, src_domain, G, C, get_mnist_mnistm, domains, get_acc_dict
        )

    @torch.no_grad()
    @unittest.skipIf(not RUN_PRETRAINED_SCORES_TESTS, skip_reason)
    def test_office31(self):
        dataset_name = "office31"
        domains = ["amazon", "dslr", "webcam"]
        for src_domain in domains:
            check_if_keys_exist(dataset_name, src_domain, domains)

        G = office31G(pretrained=True, map_location=TEST_DEVICE)
        for src_domain in domains:
            C = office31C(domain=src_domain, pretrained=True, map_location=TEST_DEVICE)
            self.helper(
                dataset_name, src_domain, G, C, get_office31, domains, get_acc_dict
            )

    @torch.no_grad()
    @unittest.skipIf(not RUN_PRETRAINED_SCORES_TESTS, skip_reason)
    def test_officehome(self):
        dataset_name = "officehome"
        domains = ["art", "clipart", "product", "real"]
        for src_domain in domains:
            check_if_keys_exist(dataset_name, src_domain, domains)

        G = officehomeG(pretrained=True, map_location=TEST_DEVICE)
        for src_domain in domains:
            C = officehomeC(
                domain=src_domain, pretrained=True, map_location=TEST_DEVICE
            )
            self.helper(
                dataset_name, src_domain, G, C, get_officehome, domains, get_acc_dict
            )

    @torch.no_grad()
    @unittest.skipIf(not RUN_PRETRAINED_SCORES_TESTS, skip_reason)
    def test_domainnet126(self):
        dataset_name = "domainnet126"
        domains = ["clipart", "painting", "real", "sketch"]
        for src_domain in domains:
            check_if_keys_exist(dataset_name, src_domain, domains)

        for src_domain in domains:
            G = domainnet126G(
                domain=src_domain, pretrained=True, map_location=TEST_DEVICE
            )
            C = domainnet126C(
                domain=src_domain, pretrained=True, map_location=TEST_DEVICE
            )
            self.helper(
                dataset_name, src_domain, G, C, get_domainnet126, domains, get_acc_dict
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--print_output_instead_of_asserting", action="store_true")
    parser.add_argument("--hide_progress_bars", action="store_true")
    args, unittest_args = parser.parse_known_args()
    if args.hide_progress_bars:
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    for k in vars(args):
        setattr(TestPretrainedScores, k, getattr(args, k))

    sys.argv[1:] = unittest_args
    unittest.main()
