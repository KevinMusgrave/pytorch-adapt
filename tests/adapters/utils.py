import torch
from torchvision import datasets, models, transforms

from pytorch_adapt.containers import Models
from pytorch_adapt.datasets import (
    CombinedSourceAndTargetDataset,
    SourceDataset,
    TargetDataset,
)
from pytorch_adapt.models import Classifier, Discriminator


class FakeDataForAdaptation(torch.utils.data.Dataset):
    def __init__(self, size):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.FakeData(
            size=size, image_size=(3, 224, 224), transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_source_model():
    source_model = models.resnet18()
    source_classifier = Classifier(in_size=source_model.fc.in_features, num_classes=10)
    source_model.fc = torch.nn.Identity()
    return source_model, source_classifier


def get_datasets():
    src_train = SourceDataset(FakeDataForAdaptation(500))
    target_train = TargetDataset(FakeDataForAdaptation(300))
    return {
        "train": CombinedSourceAndTargetDataset(src_train, target_train),
        "src_train": src_train,
        "src_val": SourceDataset(FakeDataForAdaptation(200)),
        "target_train": target_train,
        "target_val": TargetDataset(FakeDataForAdaptation(100)),
    }


def get_gcd():
    source_model, source_classifier = get_source_model()
    discriminator = Discriminator(in_size=source_classifier.net[0].in_features)
    return Models(
        {
            "G": source_model,
            "C": source_classifier,
            "D": discriminator,
        }
    )
