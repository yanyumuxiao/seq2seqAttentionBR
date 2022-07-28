import pickle
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import Config
from datasets import DataInput, TestDataInput
from model import Seq2Seq
from utils import init_seed, load_data, epoch_time


def main(config):
    init_seed(config.seed)
    device = config.device

    train_set, val_set, test_set, bundle_map, adj = load_data(config)

    adj = adj.to(device)

    model = Seq2Seq(config, adj).to(device)

    loss_func = nn.CrossEntropyLoss(ignore_index=config.pad)
    optimizer = optim.SGD(model.parameters(), lr=1)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(config.epoch):
        print('--------------epoch: {}--------------'.format(epoch))
        # model.train()
        # train_loss = train(config, model, train_set, bundle_map, loss_func, optimizer, device)
        #
        model.eval()
        auc = evaluate(config, model, val_set, bundle_map, loss_func, device)
        test_loss, pre, div = test(config, model, test_set, bundle_map, loss_func, device)


def train(config, model, train, bundle_map, loss_func, optimizer, device):
    start_time = time.time()
    epoch_loss = 0
    print('start train:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for i, uij in tqdm(DataInput(config, train, bundle_map)):
        inputs, pos, neg = uij
        optimizer.zero_grad()
        inputs = inputs.to(device)
        pos = pos.to(device)

        output, context = model(inputs, pos, teacher_forcing_ratio=config.teacher_forcing_ratio)
        output = output[1:].view(-1, output.shape[-1])
        pos = pos.view(-1)

        loss = loss_func(output, pos) + config.reg_rate * torch.sum(context ** 2) / 2
        # loss = torch.sum(loss) / torch.sum(loss != 0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪，防止梯度爆炸。clip：梯度阈值
        optimizer.step()
        epoch_loss += loss.item()

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('\tTrain Loss: {:.4f}\tCost: {}m {}s'.format(epoch_loss / (i + 1), epoch_mins, epoch_secs))

    # return epoch_loss / (i + 1)


def evaluate(config, model, val, bundle_map, loss_func, device):
    start_time = time.time()
    auc = 0.0
    with torch.no_grad():
        print('start eval:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for i, uij in tqdm(DataInput(config, val, bundle_map)):
            inputs, pos, neg = uij
            inputs = inputs.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_output, _ = model(inputs, pos, teacher_forcing_ratio=0)
            pos_output = pos_output[1:].transpose(0, 1)
            neg_output, _ = model(inputs, neg, teacher_forcing_ratio=0)
            neg_output = neg_output[1:].transpose(0, 1)

            # pos_output = pos_output[1:].view(-1, pos_output.shape[-1])
            # pos = pos.view(-1)
            for pos_logit, neg_logit, target, negative in zip(pos_output, neg_output, pos, neg):
                pos_loss = loss_func(pos_logit, target)
                neg_loss = loss_func(neg_logit, negative)
                # 计算batch内的每一项的损失
                if pos_loss < neg_loss:
                    auc += 1
    auc /= len(val)
    # pos_output = pos_output[1:].view(-1, pos_output.shape[-1])
    # pos = pos.view(-1)
    # pos_loss = loss_func(pos_output, pos)
    # pos_loss = torch.sum(pos_loss.reshape((inputs.shape[0], -1)), -1) / torch.sum(pos_loss != 0)

    # neg_output = model(inputs, neg, teacher_forcing_ratio=0)
    # neg_output = neg_output[1:].view(-1, neg_output.shape[-1])
    # neg = neg.view(-1)
    # neg_loss = loss_func(neg_output, neg)
    # neg_loss = torch.mean(neg_loss.reshape((inputs.shape[0], -1)), -1)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print('\tAUC: {:.4f}\tCost: {}m {}s'.format(auc, epoch_mins, epoch_secs))

    return auc


def test(config, model, test, bundle_map, loss_func, device):
    start_time = time.time()
    epoch_loss = 0
    with torch.no_grad():
        print('start test:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for i, uij in tqdm(TestDataInput(config, test, bundle_map)):
            inputs, pos = uij
            inputs = inputs.to(device)
            pos = pos.to(device)

            output, _ = model(inputs, pos, teacher_forcing_ratio=0)
            output = output[1:].view(-1, output.shape[-1])
            pos = pos.view(-1)

            loss = loss_func(output, pos)
            epoch_loss += loss.item()
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    pre = 1
    div = 1
    print('\tTest Loss: {:.4f}\tpre: {:.4f}\tdiv: {:.4f}\tCost: {}m {}s'
          .format(epoch_loss / (i + 1), pre, div, epoch_mins, epoch_secs))

    return epoch_loss / (i + 1), pre, div


if __name__ == '__main__':
    main(Config())
