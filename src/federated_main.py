import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFlower, RiverNDMLP
from src.utils import get_dataset, average_weights, exp_details, fed_avg
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar()
            global_model = models.resnet50(pretrained=True).to(device)
            for param in global_model.parameters():
                param.requires_grad = False

            num_ftrs = global_model.fc.in_features
            global_model.fc = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(num_ftrs, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10),
            ).to(device)
        elif args.dataset == 'flower':
            global_model = CNNFlower(args=args)

    elif args.model == 'mlp':
        if args.dataset == 'river2d':
            global_model = RiverNDMLP(n_features=2)
        else:
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    max_acc = 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses , local_datapoints = [], [], dict()
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        dataset_size_per_client = [len(user_groups[i]) for i in idxs_users]
        i = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_datapoints[i] = len(user_groups[idx])
            i += 1

        # global_weights = average_weights(local_weights, local_datapoints)
        global_weights = fed_avg(local_weights, dataset_size_per_client)

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        list_acc, list_loss = [], []

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in DataLoader(test_dataset, batch_size=64, shuffle=False):
                # images, labels = images.to(self.device), labels.to(self.device)
                images = images.to('cuda:0')
                labels = labels.to('cuda:0')
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False).to('cuda:0')
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                for i, label in enumerate(labels):
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

        if max_acc < acc:
            max_acc = acc
            torch.save(global_model, "FrozenModels/DirichletFederatedCifar")

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

# PLOTTING (optional)
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
#
# # Plot Loss curve
# plt.figure()
# plt.title('Training Loss vs Communication rounds')
# plt.plot(range(len(train_loss)), train_loss, color='r')
# plt.ylabel('Training loss')
# plt.xlabel('Communication Rounds')
# plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
#             format(args.dataset, args.model, args.epochs, args.frac,
#                    args.iid, args.local_ep, args.local_bs))
#
# # Plot Average Accuracy vs Communication rounds
# plt.figure()
# plt.title('Average Accuracy vs Communication rounds')
# plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
# plt.ylabel('Average Accuracy')
# plt.xlabel('Communication Rounds')
# plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
#             format(args.dataset, args.model, args.epochs, args.frac,
#                    args.iid, args.local_ep, args.local_bs))
