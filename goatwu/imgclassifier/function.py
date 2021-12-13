import os
import sys
sys.path.append("..")
import goatwu.basicfunc
import goatwu.imgclassifier.dataset.loader as loader
import goatwu.imgclassifier.dataset.trans as trans

import numpy as np
import pandas as pd
import copy
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from tqdm import tqdm
import ttach as tta

function = sys.modules[__name__]

# ---------------------------------- initialize functions -------------------------------------------

def xavier_init(net):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)


def pretrained_init(net):
    nn.init.xavier_uniform_(net.fc.weight);
    
# ---------------------------------- optimizer ------------------------------------------

def sgd_default(net, lr=0.1, momentum=0.9, weight_decay=5e-4):
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

def adamw_default(net, lr=1e-4, wd=1e-3):
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

def sgd_train_fine_tuning(net, lr=1e-4, wd=1e-3, param_group=True):
    if param_group:
        params_1x = [param for name, param in net.named_parameters() 
                     if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': net.fc.parameters(),
                                      'lr': learning_rate*10}],
                                    lr=learning_rate, weight_decay=wd)
    else:
        torch.optim.SGD(net.parameters(), lr=learning_rate,
                        weight_decay=wd)
    return optimizer

def adam_train_fine_tuning(net, learning_rate=1e-4, wd=1e-3, param_group=True):
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.Adam([{'params': params_1x}, 
                                      {'params': net.fc.parameters(),
                                       'lr': learning_rate*10}], 
                                     lr=learning_rate, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                      weight_decay=wd)  
    return optimizer

# ---------------------------------- evaluate accuracy -------------------------------------------

def evaluate_accuracy(net, dataloader, device):
    net.eval()
    valid_accs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, labels = batch
            labels = labels.to(device)
            predicts = net(imgs.to(device))
            predicts = predicts.argmax(dim=1)
            cmp = predicts.type(labels.dtype) == labels
            acc = float(cmp.type(labels.dtype).sum()) / float(len(labels))
            valid_accs.append(acc)
    valid_acc = np.sum(valid_accs) / len(valid_accs)
    return valid_acc

# ---------------------------------- add tensorboard -------------------------------------------

def add_info_to_tensorboard(writer, epoch, train_loss, train_acc, valid_acc, scheduler, optimizer, folder_id):
    if folder_id is None:
        writer.add_scalar("Loss/train", train_loss, epoch+1)
        writer.add_scalar("Accuracy/train", train_acc, epoch+1)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch+1)
        writer.add_scalars("All-In-One", {'train loss': train_loss, 
                                          'train accuracy': train_acc, 
                                          'valid accuracy': valid_acc}, epoch+1)
        if scheduler is not None:
            writer.add_scalar("learning-rate", scheduler.get_last_lr()[0], epoch+1)
    else:
        writer.add_scalar("Folder-{}/Loss/train".format(folder_id), train_loss, epoch+1)
        writer.add_scalar("Folder-{}/Accuracy/train".format(folder_id), train_acc, epoch+1)
        writer.add_scalar("Folder-{}/Accuracy/valid".format(folder_id), valid_acc, epoch+1)
        writer.add_scalars("Folder-{}/All-In-One".format(folder_id), 
                           {'train loss': train_loss, 
                            'train accuracy': train_acc, 
                            'valid accuracy': valid_acc}, epoch+1)
        if scheduler is not None:
            writer.add_scalar("Folder-{}/learning-rate".format(folder_id), scheduler.get_last_lr()[0], epoch+1)

    
# ---------------------------------- save net functions -------------------------------------------

def save_init_state(net, optimizer, scheduler, checkpoint_dir):
    if scheduler is not None:
        state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict(), 
                 'scheduler': scheduler.state_dict()}
    else:
        state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(checkpoint_dir, 'initial.pth'))

def save_best_state(net, best_acc, epoch, folder_id, checkpoint_dir):
    state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': epoch + 1,
    }
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if folder_id is None:
        torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))
    else:
        torch.save(state, os.path.join(checkpoint_dir, 'Folder-{}-best.pth'.format(folder_id)))
            
# ---------------------------------- train functions -------------------------------------------
    
def train_action(net, trainloader, validloader, cutmix_evaluate_loader,
                 num_epochs, optimizer, loss, device, scheduler, writer, folder_id, checkpoint_dir):
    best_acc = 0
    best_train_acc = 0
    best_train_loss = 0
    for epoch in range(num_epochs):
        net.train()
        print(f'Start training epoch {epoch+1} ...')
        train_losses = []
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            predict = net(imgs)
            l = loss(predict, labels)
            l.backward()
            optimizer.step()
            train_losses.append(l.item())
        train_loss = np.sum(train_losses) / len(train_losses)
        if cutmix_evaluate_loader is None:
            train_acc = function.evaluate_accuracy(net, trainloader, device)
        else:
            train_acc = function.evaluate_accuracy(net, cutmix_evaluate_loader, device)
        valid_acc = function.evaluate_accuracy(net, validloader, device)
        if scheduler is not None:
            scheduler.step()
        if writer is not None:
            add_info_to_tensorboard(writer, epoch, train_loss, train_acc, valid_acc, scheduler, optimizer, folder_id)
            
        print("epoch {}:\ttrain_acc: {},\ttest_acc: {},\tloss: {}".format(epoch+1, train_acc, valid_acc, train_loss))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_train_acc = train_acc
            best_train_loss = train_loss
            save_best_state(net, best_acc, epoch, folder_id, checkpoint_dir)

    return best_acc, best_train_acc, best_train_loss


def train(net, trainset, validset, num_epochs, batch_size, optimizer, loss, 
          use_tensorboard=True, init_func=function.xavier_init, 
          train_loader_func=loader.default_train_loader,
          valid_loader_func=loader.default_valid_loader,
          use_cutmix=False, split_ratio=0.1,
          device='cpu', scheduler=None, kfold=None, checkpoint_dir=None):
    init_func(net)
    net.to(device)
    writer = SummaryWriter(os.path.join('./log', net.__class__.__name__)) if use_tensorboard else None
    checkpoint_dir = os.path.join('./checkpoint', net.__class__.__name__) if checkpoint_dir is None else checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if kfold is None:
        if validset is None: # 没有给定验证集，手动分离
            trainset, validset = loader.generate_train_valid_set(trainset, split_ratio)
        trainloader = train_loader_func(trainset, batch_size)
        validloader = valid_loader_func(validset, batch_size)
        cutmix_evaluate_loader = loader.default_train_loader(trainset, batch_size) if use_cutmix else None
        print("=" * 25 + " start training {} epochs ".format(num_epochs) + "=" * 25 + "\n")
        acc, train_acc, train_loss = train_action(net=net, trainloader=trainloader, validloader=validloader, 
                                                  cutmix_evaluate_loader=cutmix_evaluate_loader,
                                                  num_epochs=num_epochs, optimizer=optimizer, loss=loss, 
                                                  device=device, scheduler=scheduler, writer=writer, 
                                                  folder_id=None, checkpoint_dir=checkpoint_dir)
        print("\nThe Best Result: train_acc: {},\ttest_acc: {},\tloss: {}\n".format(train_acc, acc, train_loss))
        print("=" * 25 + " end training {} epochs ".format(num_epochs) + "=" * 25 + "\n")
            
    elif validset is None: # 使用K折交叉验证，不应当给定验证集
        # 每个fold需要从头开始训练，因此保存初始的网络和优化器
        save_init_state(net, optimizer, scheduler, checkpoint_dir)
        result = []
        print("=" * 25 + " start training k-fold " + "=" * 25 + "\n")
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(trainset)):
            # 读入初始的网络和优化器
            initial_ckpt = torch.load(os.path.join(checkpoint_dir, 'initial.pth'))
            net.load_state_dict(initial_ckpt['net'])
            optimizer.load_state_dict(initial_ckpt['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(initial_ckpt['scheduler'])
            print("-" * 15 + " start training fold-{} in {} epochs ".format(fold, num_epochs) + "-" * 15 + "\n")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
            trainloader = train_loader_func(trainset, batch_size, sampler=train_subsampler)
            validloader = valid_loader_func(trainset, batch_size, sampler=valid_subsampler)
            cutmix_evaluate_loader = loader.default_train_loader(trainset, batch_size, sampler=train_subsampler) if use_cutmix else None
            acc, train_acc, train_loss = train_action(net=net, trainloader=trainloader, validloader=validloader, 
                                                      cutmix_evaluate_loader=cutmix_evaluate_loader,
                                                      num_epochs=num_epochs, optimizer=optimizer, loss=loss, 
                                                      device=device, scheduler=scheduler, writer=writer, 
                                                      folder_id=fold, checkpoint_dir=checkpoint_dir)
            print("\nThe Best Result: train_acc: {},\ttest_acc: {},\tloss: {}\n".format(train_acc, acc, train_loss))
            print("-" * 15 + " end training fold-{} in {} epochs ".format(fold, num_epochs) + "-" * 15 + "\n")
            result.append(acc)
        total = 0.0
        for value in result:
            total += value
        print(f'Average accuracy: {total/len(result)} ')
        print("=" * 25 + " end training k-fold " + "=" * 25 + "\n")
    else:
        raise("validset should not be given when using k-fold.")
        
# ---------------------------------- predict functions -------------------------------------------

def generate_single_result(net, testloader, data_path, result_name, num2cls,
                           device, checkpoint, use_tta, tta_height):
    net.to(device)
    ckpt = torch.load(checkpoint)
    net.load_state_dict(ckpt['net'])
    net.eval()
    net = trans.ttafunc(net, tta_height) if use_tta else net
    predictions = []
    with torch.no_grad():
        for batch in tqdm(testloader):
            imgs = batch
            predicts = net(imgs.to(device))
            predictions.extend(predicts.argmax(dim=1).cpu().numpy().tolist())
    labels = []
    for num in predictions:
        labels.append(num2cls[num])
    data = pd.read_csv(data_path)
    data['label'] = pd.Series(labels)
    submission = pd.concat([data['image'], data['label']], axis=1)
    submission.to_csv(result_name, index=False)
        

def predict(net, testset, batch_size, test_loader_func=loader.default_test_loader,
            data_path=None, device='cpu', kfold=None, use_tta=False, tta_height=0,
            checkpoint_dir=None, submission_dir=None):
    checkpoint_dir = os.path.join('./checkpoint', net.__class__.__name__) if checkpoint_dir is None else checkpoint_dir
    submission_dir = os.path.join('./submission', net.__class__.__name__) if submission_dir is None else submission_dir
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    testloader = test_loader_func(testset, batch_size)
    if kfold is None:
        result_name = os.path.join(submission_dir, 'submission.csv')
        checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        generate_single_result(net, testloader, data_path, result_name, testset.num2cls, 
                               device, checkpoint, use_tta, tta_height)
    else:
        for fold in range(kfold):
            result_name = os.path.join(submission_dir, 'submission-fold-{}.csv'.format(fold))
            checkpoint = os.path.join(checkpoint_dir, 'Folder-{}-best.pth'.format(fold))
            generate_single_result(net, testloader, data_path, result_name, testset.num2cls, 
                                   device, checkpoint, use_tta, tta_height)
            

def voting_kfold(submission_dir, kfold, cls2num, num2cls, name=None):
    df = []
    list_num_label = []
    for fold in range(kfold):
        csvpath = os.path.join(submission_dir, 'submission-fold-{}.csv'.format(fold))
        df.append(pd.read_csv(csvpath))
    for fold in range(kfold):
        list_num_label_tmp = []
        for i in df[fold]['label']:
            list_num_label_tmp.append(cls2num[i])
        df[fold]['num_label_{}'.format(fold)] = list_num_label_tmp
    df_all = df[0].copy()
    df_all.drop(['label'], axis=1, inplace=True)
    for fold in range(1, kfold):
        df_all['num_label_{}'.format(fold)] = df[fold]['num_label_{}'.format(fold)]
    df_all_transpose = df_all.copy().drop(['image'],axis=1).transpose()
    df_mode = df_all_transpose.mode().transpose()
    voting_class = []
    for each in df_mode[0]:
        voting_class.append(num2cls[each])
    df_all['label'] = voting_class
    df_submission = df_all[['image','label']].copy()
    submission_name = os.path.join(submission_dir, name) if name is not None else os.path.join(submission_dir, 'submission.csv')
    df_submission.to_csv(submission_name, index=False)
    
    
def voting_models(cls2num, num2cls, submission_name, *args):
    df = []
    for i, csvpath in enumerate(args):
        df.append(pd.read_csv(csvpath))
    for j in range(len(df)):
        list_num_label_tmp = []
        for i in df[j]['label']:
            list_num_label_tmp.append(cls2num[i])
        df[j]['num_label_{}'.format(j)] = list_num_label_tmp
    df_all = df[0].copy()
    df_all.drop(['label'], axis=1, inplace=True)
    for j in range(1, len(df)):
        df_all['num_label_{}'.format(i)] = df[j]['num_label_{}'.format(j)]
    df_all_transpose = df_all.copy().drop(['image'],axis=1).transpose()
    df_mode = df_all_transpose.mode().transpose()
    voting_class = []
    for each in df_mode[0]:
        voting_class.append(num2cls[each])
    df_all['label'] = voting_class
    df_submission = df_all[['image','label']].copy()
    df_submission.to_csv(submission_name, index=False)
    
