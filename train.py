import numpy as np
import pandas as pd
import torch
import models

from preprocess import traverse
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def draw(train_loss, val_loss, val_acc):
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, c='r', label='train loss')
    plt.plot(val_loss, c='y', label='val loss')

    plt.subplot(2, 1, 2)
    plt.plot(val_acc, c='g', label='val acc')
    plt.show()


def accuracy(Y_hat, Y, averaged=True):
    Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = torch.argmax(Y_hat, dim=1).type(Y.dtype)
    # print(Y_hat.shape, preds.shape, Y.shape)
    compare = (preds == Y).type(torch.float32)
    return torch.mean(compare) if averaged else compare


def split_dataset(tensors, save=False, split=None, load=False):
    if split is None:
        split = [0.8, 0.1, 0.1]
    data_num = tensors[0].shape[0]

    train_num = int(split[0] * data_num)
    val_num = int(split[1] * data_num)
    test_num = data_num - train_num - val_num

    dataset = torch.utils.data.TensorDataset(*tensors)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_num, val_num, test_num])

    dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
                   for dataset in [train_dataset, val_dataset, test_dataset]]
    if save:
        torch.save(dataloaders, 'dataset.dat')
    return dataloaders


if __name__ == '__main__':
    root_path = r'F:\data\plane\A'  # 数据路径
    max_epoch = 100
    num_feature = 6
    batch_size = 32

    tensors = traverse(root_path)
    data_loaders = split_dataset(tensors, load=False)


    model = models.GRUModel(num_class=14, num_feature=num_feature).cuda()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(max_epoch):
        batch_train_loss = []
        batch_val_loss = []
        batch_val_acc = []
        model.train()
        for X, y in data_loaders[0]:
            X = X.float().cuda()
            y = y.cuda()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_train_loss.append(loss)
            # print(model.state_dict())
        model.eval()
        for X, y in data_loaders[1]:
            X = X.float().cuda()
            y = y.cuda()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat, y)
            batch_val_acc.append(acc)
            batch_val_loss.append(loss)
        val = [torch.mean(torch.tensor(batch_train_loss)), torch.mean(torch.tensor(batch_val_loss)),
               torch.mean(torch.tensor(batch_val_acc))]
        print(f'epoch:{epoch} train loss:{val[0]}, 'f'val loss:{val[1]}, 'f'val acc:{val[2]}')
        train_loss.append(val[0])
        val_loss.append(val[1])
        val_acc.append(val[2])
        if epoch % 5 == 0:
            draw(train_loss, val_loss, val_acc)
