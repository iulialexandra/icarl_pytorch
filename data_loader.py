from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image
from torchvision import datasets


class iCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        dataset = datasets.cifar.CIFAR10(root, train=train,
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
#
# from torchvision.datasets import CIFAR10
# import numpy as np
# import torch
# from PIL import Image
#
#
# class iCIFAR10(CIFAR10):
#     def __init__(self, root,
#                  classes=range(10),
#                  train=True,
#                  transform=None,
#                  target_transform=None,
#                  download=False):
#         super(iCIFAR10, self).__init__(root,
#                                        train=train,
#                                        transform=transform,
#                                        target_transform=target_transform,
#                                        download=download)
#
#         # Select subset of classes
#         if self.train:
#             train_data = []
#             train_labels = []
#
#             for i in np.arange(len(self.data)):
#                 if self.targets[i] in classes:
#                     train_data.append(self.data[i])
#                     train_labels.append(self.targets[i])
#
#             self.train_data = np.array(train_data)
#             self.train_labels = train_labels
#
#         else:
#             test_data = []
#             test_labels = []
#
#             for i in np.arange(len(self.data)):
#                 if self.targets[i] in classes:
#                     test_data.append(self.data[i])
#                     test_labels.append(self.targets[i])
#
#             self.test_data = np.array(test_data)
#             self.test_labels = test_labels
#
#     def __getitem__(self, index):
#         if self.train:
#             img, target = self.train_data[index], self.train_labels[index]
#         else:
#             img, target = self.test_data[index], self.test_labels[index]
#
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return index, img, target
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)
#
#     def get_image_class(self, label):
#         return self.train_data[np.array(self.train_labels) == label]
#
#     def append(self, images, labels):
#         """Append dataset with images and labels
#
#         Args:
#             images: Tensor of shape (N, C, H, W)
#             labels: list of labels
#         """
#
#         self.train_data = np.concatenate((self.train_data, images), axis=0)
#         self.train_labels = self.train_labels + labels
#
#
# class iCIFAR100(iCIFAR10):
#     base_folder = 'cifar-100-python'
#     url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     filename = "cifar-100-python.tar.gz"
#     tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
#     train_list = [
#         ['train', '16019d7e3df5f24257cddd939b257f8d'],
#     ]
#     test_list = [
#         ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
#     ]
