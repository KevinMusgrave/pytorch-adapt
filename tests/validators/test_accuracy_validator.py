import unittest

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)

from pytorch_adapt.adapters import MultiLabelClassifier
from pytorch_adapt.frameworks.ignite import (
    Ignite,
    IgniteMultiLabelClassification,
    IgnitePredsAsFeatures,
)
from pytorch_adapt.validators import AccuracyValidator, APValidator, ScoreHistory

from .utils import test_with_ignite_framework


class TestAccuracyValidator(unittest.TestCase):
    def test_accuracy_validator(self):
        def correct_score_fn(logits, labels, torchmetric_kwargs):
            np_argmax = np.argmax(logits.numpy(), axis=1)
            sklearn_score1 = accuracy_score(labels.numpy(), np_argmax)
            sklearn_score2 = balanced_accuracy_score(labels.numpy(), np_argmax)
            self.assertTrue(sklearn_score1 != sklearn_score2)
            return sklearn_score1 if torchmetric_kwargs is None else sklearn_score2

        torchmetric_kwargs_list = [None, {"average": "macro", "num_classes": 10}]
        self.helper(
            AccuracyValidator, torchmetric_kwargs_list, correct_score_fn, "logits"
        )
        self.helper(
            AccuracyValidator, torchmetric_kwargs_list, correct_score_fn, "preds"
        )

    def test_ap_validator(self):
        def correct_score_fn(preds, labels, torchmetric_kwargs):
            average = torchmetric_kwargs.pop("average", None)
            average_kwarg = {} if average is None else {"average": average}
            return average_precision_score(
                labels.numpy(), preds.numpy(), **average_kwarg
            )

        # leaving out num_classes is not allowed
        for torchmetric_kwargs_list in [
            [None],
            [{"average": "macro"}],
            {"average": "micro"},
        ]:
            with self.assertRaises(ValueError):
                self.helper(
                    APValidator,
                    torchmetric_kwargs_list,
                    correct_score_fn,
                    "preds",
                    multilabel=True,
                )

        torchmetric_kwargs_list = [
            {"num_classes": 10},
            {"average": "micro", "num_classes": 10},
            {"average": "macro", "num_classes": 10},
        ]

        # "logits" not allowed
        with self.assertRaises(ValueError):
            self.helper(
                APValidator,
                torchmetric_kwargs_list,
                correct_score_fn,
                "logits",
                multilabel=True,
            )
        self.helper(
            APValidator,
            torchmetric_kwargs_list,
            correct_score_fn,
            "preds",
            multilabel=True,
        )

    def helper(
        self,
        validator_cls,
        torchmetric_kwargs_list,
        correct_score_fn,
        layer,
        multilabel=False,
    ):
        dataset_size = 1000
        ignore_epoch = 0

        for start in [-1, 0, 1]:
            for step in [1, 2]:
                for torchmetric_kwargs in torchmetric_kwargs_list:
                    validator = ScoreHistory(
                        validator_cls(
                            layer=layer, torchmetric_kwargs=torchmetric_kwargs
                        ),
                        ignore_epoch=ignore_epoch,
                    )
                    correct_scores = []
                    for i, epoch in enumerate(range(start, 5, step)):
                        logits = torch.randn(dataset_size, 10)
                        if multilabel:
                            labels = torch.randint(0, 2, (dataset_size, 10))
                            logits = torch.sigmoid(logits)
                        else:
                            labels = torch.randint(0, 10, (dataset_size,))
                            labels[:900] = 0  # make it unbalanced
                            if layer == "preds":
                                logits = torch.softmax(logits, dim=1)
                        score = validator(
                            epoch=epoch, src_val={"labels": labels, layer: logits}
                        )

                        correct_score = correct_score_fn(
                            logits, labels, torchmetric_kwargs
                        )

                        if epoch != ignore_epoch:
                            correct_scores.append(correct_score)

                        if epoch != ignore_epoch:
                            self.assertTrue(
                                validator.score_history_ignore_epoch[validator.best_idx]
                                == validator.best_score
                            )
                            self.assertTrue(
                                validator.epochs_ignore_epoch[validator.best_idx]
                                == validator.best_epoch
                            )
                            self.assertTrue(
                                np.isclose(validator.best_score, max(correct_scores))
                            )
                            self.assertTrue(
                                np.isclose(
                                    validator.best_idx, np.argmax(correct_scores)
                                )
                            )
                        elif i == 0 and epoch == ignore_epoch:
                            self.assertTrue(validator.best_epoch is None)
                            self.assertTrue(validator.best_score is None)

                        self.assertTrue(np.isclose(score, correct_score))
                        self.assertTrue(validator.latest_score == score)


class TestAPValidatorWithIgnite(unittest.TestCase):
    def test_ap_validator(self):

        for num_classes in [5, 13, 19]:
            for average in ["micro", "macro"]:
                for ignite_cls_list in [
                    [Ignite],
                    [IgnitePredsAsFeatures],
                    [IgniteMultiLabelClassification],
                ]:

                    def assertion_fn(logits, labels, score):
                        correct_score = average_precision_score(
                            labels["src_val"].cpu().numpy(),
                            torch.sigmoid(logits["src_val"]).cpu().numpy(),
                            average=average,
                        )

                        if ignite_cls_list in [[Ignite], [IgnitePredsAsFeatures]]:
                            self.assertNotAlmostEqual(score, correct_score, places=6)
                        else:
                            self.assertAlmostEqual(score, correct_score, places=6)

                    test_with_ignite_framework(
                        APValidator(
                            torchmetric_kwargs={
                                "num_classes": num_classes,
                                "average": average,
                            }
                        ),
                        assertion_fn,
                        num_classes,
                        multilabel=True,
                        adapter_cls=MultiLabelClassifier,
                        ignite_cls_list=ignite_cls_list,
                    )
