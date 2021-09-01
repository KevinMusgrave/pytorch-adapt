import os
from collections import defaultdict

import torch
import tqdm
from torchvision import transforms as torch_transforms

skip_reason = "RUN_DATASET_TESTS is False"


def simple_transform():
    return torch_transforms.Compose(
        [
            torch_transforms.Resize((2, 2)),
            torch_transforms.ToTensor(),
        ]
    )


def loop_through_dataset(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)
    for _ in tqdm.tqdm(dataloader):
        pass


def no_duplicates_and_correct_number_of_classes(cls, d, num_classes, dataset_folder):
    cls.assertTrue(len(set(d.img_paths)) == len(d.img_paths))
    cls.assertTrue(len(set(d.labels)) == num_classes)
    assert_classnames_match_labels(cls, d, dataset_folder)


def check_full(cls, num_classes, domains, full_class, dataset_folder):
    transform = simple_transform()
    for domain in domains:
        d = full_class(dataset_folder, domain, transform)
        no_duplicates_and_correct_number_of_classes(cls, d, num_classes, dataset_folder)
        loop_through_dataset(d)


def check_train_test_disjoint(cls, num_classes, train, test, dataset_folder):
    for d in [train, test]:
        no_duplicates_and_correct_number_of_classes(cls, d, num_classes, dataset_folder)

    cls.assertTrue(set(train.img_paths).isdisjoint(set(test.img_paths)))
    return torch.utils.data.ConcatDataset([train, test])


def check_train_test_matches_full(
    cls, num_classes, domains, full_class, sub_class, dataset_folder
):
    transform = simple_transform()
    for domain in domains:
        full = full_class(dataset_folder, domain, transform)
        train = sub_class(dataset_folder, domain, True, transform)
        test = sub_class(dataset_folder, domain, False, transform)
        dataset = check_train_test_disjoint(
            cls, num_classes, train, test, dataset_folder
        )
        cls.assertTrue(len(dataset) == len(full))
        loop_through_dataset(dataset)


def assert_classnames_match_labels(cls, d, dataset_folder):
    classes = [x.split(os.sep)[-2] for x in d.img_paths]
    classes_to_labels = defaultdict(list)
    for i, classname in enumerate(classes):
        classes_to_labels[classname].append(d.labels[i])
    cls.assertTrue(all(len(set(x)) == 1 for x in classes_to_labels.values()))
