import datetime
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def initialize_logger(output_dir, print_to_console):
    """initializes loggers for debug and error, prints to file

    Args:
        output_dir: the directory of the logger file
        print_to_console: flag, whether to print logging info to console
    """
    logger = logging.getLogger("iCaRL")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  "%d-%m-%Y %H:%M:%S")
    # Setup console logging
    if print_to_console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Setup file logging
    fh = logging.FileHandler(os.path.join(output_dir, "log.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def make_results_dir(save_path, inc_epocs, seed, exemplars):
    """Makes one folder for the results using the current date and time
     and initializes the logger.
    """
    now = datetime.datetime.now()
    date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                         now.second, now.microsecond)
    results_path = os.path.join(save_path,
                                date + "_seed_{}".format(seed) + "_{}_epochs".format(inc_epocs)
                                + "_{}_exemplars".format(exemplars))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # else:
    #     path_list = self.base_chkpt.split("/")[: -3]
    #     self.results_path = os.path.join("/", *path_list)
    return results_path


def str_to_class(str):
    """Gets a class when given its name.

    Args:
        str: the name of the class to retrieve
    """
    return getattr(sys.modules[__name__], str)


def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    import yaml
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                print("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def ims_to_comet(experiment, images, labels, type):
    for i, image in enumerate(images):
        experiment.log_image(image.numpy(), name="{} class image {}, class:{}".format(type, i, labels[i]),
                             image_channels="first")


def weights_to_comet(experiment, model, step, hist_name):
    weights = []
    for name in model.named_parameters():
        if 'weight' in name[0] and 'bias' not in name[0]:
            weights.extend(np.ravel(name[1].detach().cpu().numpy()))
    experiment.log_histogram_3d(np.ravel(weights), name=hist_name, step=step)
