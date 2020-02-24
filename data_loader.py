from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image
from torchvision import datasets


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, root,
                 dataset,
                 classes,
                 train=True,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if dataset == "cifar10":
            dataset = datasets.cifar.CIFAR10(root, train=train,
                                             download=True,
                                             transform=None,
                                             target_transform=None)
        elif dataset == "cifar100":
            dataset = datasets.cifar.CIFAR100(root, train=train,
                                              download=True,
                                              transform=None,
                                              target_transform=None)
        targets = np.array(dataset.targets)

        data = []
        labels = []
        for i in np.arange(len(dataset.data)):
            if targets[i] in classes:
                data.append(dataset.data[i])
                labels.append(targets[i])

        self.data = np.array(data)
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.data = np.concatenate((self.data, images), axis=0)
        self.labels = self.labels + labels
