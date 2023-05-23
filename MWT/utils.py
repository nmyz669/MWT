import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


import pandas as pd
import random


class Getdatalist:
    def __init__(self, csv_root):
        self.path_list = []
        self.label_list = []
        self.csv_root = csv_root

    def get_sheet_path(self, col1, col2, col1_total, col2_total, num):
        self.mycsv = pd.read_csv(self.csv_root, dtype=str)
        for i in range(0, num):
            k = random.randint(0, col1_total - 1)
            j = random.randint(0, col2_total - 1)
            self.path_list.append(self.mycsv.iloc[[k], [col1]].values[0][0])
            self.label_list.append(0)
            self.path_list.append(self.mycsv.iloc[[j], [col2]].values[0][0])
            self.label_list.append(1)
        return self.path_list, self.label_list

    def get_batch_path(self):
        self.get_sheet_path(0, 1, 45118, 22559, 160)
        self.get_sheet_path(2, 3, 15724, 7862, 60)
        self.get_sheet_path(4, 5, 3795, 1897, 10)
        self.get_sheet_path(6, 7, 6422, 3211, 20)
        self.get_sheet_path(8, 9, 3928, 1964, 10)
        self.get_sheet_path(10, 11, 12483, 6241, 40)
        self.get_sheet_path(12, 13, 2158, 1079, 10)
        self.get_sheet_path(14, 15, 3513, 1756, 10)
        return self.path_list, self.label_list

    def get_epoch_path(self, step_num):
        for i in range(0, step_num):
            self.get_batch_path()
        return self.path_list, self.label_list

    def get_batch_path_val(self):
        self.get_sheet_path(0, 1, 5639, 2819, 160)
        self.get_sheet_path(2, 3, 1965, 982, 60)
        self.get_sheet_path(4, 5, 474, 237, 10)
        self.get_sheet_path(6, 7, 802, 401, 20)
        self.get_sheet_path(8, 9, 491, 245, 10)
        self.get_sheet_path(10, 11, 1560, 780, 40)
        self.get_sheet_path(12, 13, 269, 134, 10)
        self.get_sheet_path(14, 15, 439, 219, 10)
        return self.path_list, self.label_list

    def get_epoch_path_val(self, step_num):
        for i in range(0, step_num):
            self.get_batch_path_val()
        return self.path_list, self.label_list

    def get_epoch_valpath(self, col1_total, col2_total):
        self.mycsv = pd.read_csv(self.csv_root, dtype=str)
        for i in range(0, col1_total):
            self.path_list.append(self.mycsv.iloc[[i], [0]].values[0][0])
            self.label_list.append(0)
        for j in range(0, col2_total):
            self.path_list.append(self.mycsv.iloc[[j], [1]].values[0][0])
            self.label_list.append(1)
        return self.path_list, self.label_list

