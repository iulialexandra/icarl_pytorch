from comet_ml import Experiment
import torch
import torchvision.transforms as transforms
import pprint
import numpy as np
import utils
from datetime import datetime
from data_loader import CifarDataset
from model import iCaRL
from torch.autograd import Variable
from utils import *


def evaluate(model, dataset):
    total = 0.0
    correct = 0.0
    for indices, images, labels in dataset:
        images = Variable(images).cuda()
        preds = model.classify(images)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()
    acc = (100 * correct / total)
    return acc, labels, preds.data.cpu()


def main():
    # load configuration from yaml file
    cfg = load_cfg("params.yaml")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    seed = cfg['SEED']
    num_epochs = cfg['EPOCHS_PER_TASK']
    num_exemplars = cfg['K']
    cls_per_phase = cfg['NUM_CLASSES']
    total_cls_num = cfg['TOTAL_CLASSES']
    save_dir = cfg['SAVE_PATH']
    lr = cfg['INITIAL_LR']
    batch_size = cfg['BATCH_SIZE']
    dataset = cfg['DATASET']

    # log experiment to Comet
    experiment = Experiment(api_key="rADdhVM9f36nJ6poH2N9L6fw2",
                            project_name="pytorch_icarl", workspace="iulialexandra",
                            disabled=False)

    now = datetime.datetime.now()
    date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                         now.second, now.microsecond)
    experiment.set_name(date + "_seed_{}".format(seed)
                        + "_{}_epochs".format(num_epochs)
                        + "_{}_exemplars".format(num_exemplars))
    experiment.log_parameters(cfg)

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results_path = utils.make_results_dir(save_dir, num_epochs, seed, num_exemplars)
    logger = utils.initialize_logger(results_path, True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Initialize CNN
    icarl = iCaRL(2048, cls_per_phase, learning_rate=lr)
    icarl.cuda()

    experiment.set_model_graph(str(icarl))
    weights_to_comet(experiment, icarl, 0, "weights_hist")

    for inc_phase, s in enumerate(range(0, total_cls_num, cls_per_phase)):
        logger.info("Loading training examples for classes {}".format(np.arange(s,
                                                                                s + cls_per_phase)))
        train_set_new = CifarDataset(root='./data',
                                     dataset=dataset,
                                     train=True,
                                     classes=range(s, s + cls_per_phase),
                                     transform=transform_train)
        train_loader_new = torch.utils.data.DataLoader(train_set_new, batch_size=batch_size,
                                                       shuffle=True, num_workers=2)

        test_set_new = CifarDataset(root='./data',
                                    dataset=dataset,
                                    train=False,
                                    classes=range(s, s + cls_per_phase),
                                    transform=transform_test)
        test_loader_new = torch.utils.data.DataLoader(test_set_new, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)

        test_set_old = CifarDataset(root='./data',
                                    dataset=dataset,
                                    train=False,
                                    classes=range(icarl.n_known),
                                    transform=transform_test)
        test_loader_old = torch.utils.data.DataLoader(test_set_old, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)

        # Update representation via BackProp
        icarl.update_representation(experiment, train_set_new, batch_size,
                                    num_epochs, inc_phase, lr)
        # ex_per_class = num_exemplars // icarl.n_classes
        ex_per_class = num_exemplars

        # Reduce exemplar sets for known classes
        icarl.reduce_exemplar_sets(ex_per_class)

        # Construct exemplar sets for new classes
        for y in np.arange(icarl.n_known, icarl.n_classes):
            logger.info("Constructing exemplar set for class {}...".format(y))
            images = train_set_new.get_image_class(y)
            icarl.construct_exemplar_set(images, ex_per_class, transform_train)
            logger.info("Done")

        for y, P_y in enumerate(icarl.exemplar_sets):
            logger.info("Exemplar set for class {}: {}".format(y, P_y.shape))
            # show_images(P_y[:10])

        icarl.recalculate_means(transform_test)

        icarl.n_known = icarl.n_classes
        logger.info("iCaRL classes: {}".format(icarl.n_known))

        weights_to_comet(experiment, icarl, inc_phase + 1, "weights_hist")

        train_acc, lbs, prds = evaluate(icarl, test_loader_new)
        plot_images = []
        labels = []
        for i in range(10):
            _, im, lab = test_set_new.__getitem__(i)
            plot_images.append(im)
            labels.append(lab)
        ims_to_comet(experiment, plot_images, labels, "current")
        experiment.log_confusion_matrix(y_true=lbs, y_predicted=prds, title="Current classes confusion Matrix at the end"
                                                         " of incremental epoch {}".format(inc_phase))

        logger.info('Accuracy on current classes: {}'.format(train_acc))
        experiment.log_metrics({"Current accuracy": train_acc}, step=inc_phase)

        if inc_phase > 0:
            test_acc, lbs, prds = evaluate(icarl, test_loader_old)
            plot_images = []
            labels = []
            for i in range(10):
                _, im, lab = test_set_old.__getitem__(i)
                plot_images.append(im)
                labels.append(lab)
            ims_to_comet(experiment, plot_images, labels, "previous")
            logger.info('Accuracy on previous classes: {}'.format(test_acc))
            experiment.log_metrics({"Previous accuracy": test_acc}, step=inc_phase)

            experiment.log_confusion_matrix(y_true=lbs, y_predicted=prds, title="Current classes confusion Matrix at the end"
                                                             " of incremental epoch {}".format(inc_phase))

if __name__ == "__main__":
    main()
