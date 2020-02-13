import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from resnet import resnet18
import logging
from utils import ims_to_comet

logger = logging.getLogger("iCaRL")


class iCaRL(nn.Module):
    def __init__(self, feature_size, n_classes, learning_rate):
        # Network architecture
        super(iCaRL, self).__init__()
        self.feature_extractor = resnet18()
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                              feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)

        self.n_classes = n_classes
        self.n_known = 0

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        # self.optimizer = optim.SGD(self.parameters(), lr=2.0,
        #                           weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print("Computing mean of exemplars...", )
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                with torch.no_grad():
                    for ex in P_y:
                        ex = Variable(transform(Image.fromarray(ex))).cuda()
                        feature = self.feature_extractor(ex.unsqueeze(0))
                        feature = feature.squeeze()
                        feature.data = feature.data / feature.data.norm()  # Normalize
                        features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print("Done")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x)  # (batch_size, feature_size)
        for i in np.arange(feature.size(0)):  # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        return preds

    def construct_exemplar_set(self, images, m, transform):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        features = []
        for img in images:
            x = Variable(transform(Image.fromarray(img)), requires_grad=False).cuda()
            feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        for k in np.arange(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
            """
            print "Selected example", i
            print "|exemplar_mean - class_mean|:",
            print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
            #features = np.delete(features, i, axis=0)
            """

        self.exemplar_sets.append(np.array(exemplar_set))

    def reduce_exemplar_sets(self, exp_per_class):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:exp_per_class]

    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)

    def update_representation(self, experiment, dataset, batch_size, epochs, inc_phase):

        self.compute_means = True

        # Increment number of weights in final fc layer
        classes = list(set(dataset.labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print("%d new classes" % (len(new_classes)))

        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

        # Store network outputs with pre-update parameters
        previous_logits = torch.zeros(len(dataset), self.n_classes).cuda()
        for indices, images, labels in loader:
            images = Variable(images).cuda()
            indices = indices.cuda()
            net_output = torch.sigmoid(self.forward(images))
            previous_logits[indices] = net_output.data
        previous_logits = Variable(previous_logits).cuda()

        # Run network training
        optimizer = self.optimizer
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.LR_EPOCHS,
        #                                                     params.LR_DECAY)
        step = 0
        for epoch in np.arange(epochs):
            # lr_scheduler.step()
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.n_classes)
                one_hot_labels = one_hot_labels.float()
                indices = indices.cuda()

                optimizer.zero_grad()
                net_output = self.forward(images)

                if self.n_known > 0:
                    # Classification loss for new classes
                    # new_label_indices = list(np.isin(labels.cpu(), new_classes))
                    # tensor_indices = torch.tensor(new_label_indices)
                    # valid_output = net_output[tensor_indices]
                    # valid_output = valid_output[:, torch.tensor(new_classes)]
                    # valid_targets = one_hot_labels[tensor_indices]
                    # valid_targets = valid_targets[:, torch.tensor(new_classes)]
                    # loss = self.dist_loss(valid_output, valid_targets)
                    loss = self.cls_loss(net_output, labels)

                    # Distilation loss for old classes
                    q_i = previous_logits[indices]
                    # dist_loss = [self.dist_loss(net_output[:, y], q_i[:, y])
                    #              for y in range(self.n_known)]
                    # dist_loss = sum(dist_loss) #/ self.n_known
                    dist_loss = self.dist_loss(torch.sigmoid(net_output), q_i)
                    loss += dist_loss
                else:
                    loss = self.cls_loss(net_output, labels)

                loss.backward()
                optimizer.step()

                experiment.log_metric("loss_inc_phase_{}".format(inc_phase), loss.data.item(),
                                      step=step, epoch=epoch)
                step += 1

                if (i + 1) % 10 == 0:
                    logger.info('Epoch {}, Iter {} Loss: {}'.format(epoch, i, loss.data.item()))
