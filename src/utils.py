#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import *


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)


        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=True, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),  # convert images, nd arrays, or tensors to other tensors
            transforms.Normalize((0.1307,), (0.3081,))])  # 1 value per channel - 1 channel

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    # elif args.dataset == 'flower':
    #     data_dir = '../data/flower/'
    #
    #     apply_transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    #     train_dataset = datasets.Flowers102(data_dir, split='train', transform=apply_transform, download=True)
    #     test_dataset = datasets.Flowers102(data_dir, split='val', transform=apply_transform, download=True)
    #
    #     if args.iid == 1:
    #         # Sample IID user data from Mnist
    #         user_groups = iid(train_dataset, args.num_users)
    #     elif args.iid == 2:
    #         user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
    # elif args.dataset == 'imdb':
    #     data_dir = '../data/imdb/'
    #
    #     train_dataset = t_datasets.IMDB(data_dir, split='train')
    #     test_dataset = t_datasets.IMDB(data_dir, split='test')
    #     user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
    #
    #     print(train_dataset)
    #
    # elif args.dataset == 'yahoo':
    #     data_dir = '../data/yahoo/'
    #
    #     train_dataset = t_datasets.YahooAnswers(data_dir, split='train')
    #     test_dataset = t_datasets.YahooAnswers(data_dir, split='test')
    #
    #     print(train_dataset)

    return train_dataset, test_dataset, user_groups


def average_weights(w, d: dict[int, int]):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    total_local_dp = sum(d.values())
    for key in w_avg.keys():
        w_avg[key] = w[0][key] * (d[0]/total_local_dp)

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (d[i]/total_local_dp)
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def fed_avg(local_weights, dataset_size_per_client):
    avg_dict = {}
    sum_dataset = sum(dataset_size_per_client)
    for i, dictionary in enumerate(local_weights):
        for key, tensor in dictionary.items():
            if key not in avg_dict:
                avg_dict[key] = tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
            else:
                avg_dict[key] += tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
    return avg_dict


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

# test push