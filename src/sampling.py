import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

def river_iid(dataset, num_users):
    """
    Sample I.I.D. client data from a 2D River dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def split_dirichlet(dataset, num_users: int, is_cfar: bool, beta: float = 0.1) -> dict[int, [int]]:
    """
    Sample non-I.I.D client data from an arbitary dataset.
    Samples it based on this paper: 10.48550/ARXIV.1905.12022
    :param dataset: The dataset
    :param num_users: The number of clients
    :param is_cfar: Whether the dataset is cfar (bool). This is a stupid solution but is necessary since for some reason,
    their parameter name is different.
    :param beta: The beta parameter used to control the distribution spread.
    :return: dict mapping client id to idxs for training
    """
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets) if is_cfar else dataset.train_labels.numpy()
    uniq_labels = np.unique(labels)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_labels = idxs_labels.T

    assert np.shape(idxs_labels) == (len(dataset), 2)

    for label in uniq_labels:
        relevant_idxs = idxs_labels[(idxs_labels[:, 1] == label)][:,0].T
        proportions = np.random.dirichlet(np.full(num_users, beta))
        splits = split_by_ratio(relevant_idxs, proportions)
        for idx, split in enumerate(splits):
            dict_users[idx] = np.concatenate([dict_users[idx], split])

    for _ , dict_val in dict_users.items():
        if len(dict_val) < 2:
            # We just restart a split if a user isn't assigned enough samples.
            return split_dirichlet(dataset, num_users, is_cfar, beta)

    return dict_users


def split_by_ratio(arr, ratios):
    """
    Splits an np array according to some proportions, must sum to 1
    """
    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]


def cifar_iid(dataset: CIFAR10, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users
