#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import numpy as np
from src.utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFlower, RiverNDMLP
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models


if __name__ == '__main__':
    args = args_parser()
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    # device = 'cuda' if args.gpu else 'cpu'
    device = 'cuda:0'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar()
            global_model = models.resnet50(pretrained=True).to(device)
            for param in global_model.parameters():
                param.requires_grad = False

                # Replace the last fully connected layer with a new classifier
            num_ftrs = global_model.fc.in_features
            global_model.fc = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(num_ftrs, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10),
                # torch.nn.Softmax(dim=1)
            ).to(device)
        elif args.dataset == 'flower':
            global_model = CNNFlower(args=args)
    elif args.model == 'mlp':
        if args.dataset == 'river2d':
            global_model = RiverNDMLP(2)
        else:
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train(True)
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=50, shuffle=True)
    print(len(trainloader))
    # criterion = torch.nn.NLLLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        running_loss = 0.0
        running_accuracy = 0.0

        for batch_idx, (images, labels) in enumerate(trainloader):

            images, labels = images.to(device), labels.to(device)
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = global_model(images)
            loss = criterion(outputs, labels)

            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / 50
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 49:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch+1, batch_idx * len(images), len(trainloader.dataset),
                #     100. * batch_idx / len(trainloader), loss.item()))
                avg_loss = running_loss / 50
                avg_accuracy = (running_accuracy / 50) * 100
                print('Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%'.format(batch_idx + 1, avg_loss, avg_accuracy))
                running_loss = 0.0
                running_accuracy = 0.0

            batch_loss.append(loss.item())

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in testloader:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False).to(device)
                images = images.to(device)
                labels = labels.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                for i in range(5):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
