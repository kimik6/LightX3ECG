
import os, sys
import os, sys
import warnings; warnings.filterwarnings("ignore")
import pytorch_lightning; pytorch_lightning.seed_everything(22)

from tqdm import tqdm

import argparse
import random
import pandas as pd
import numpy as np
import neurokit2 as nk
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from utils import *
from sklearn.metrics import classification_report
import pickle


    
    
def train_fn(
    train_loaders, 
    model, 
    num_epochs, 
    config, 
    criterion, 
    optimizer, 
    scheduler = None, 
    save_ckp_dir = "./", 
    training_verbose = True, 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids = config["device_ids"])
    all_logs = []  # Create an empty list to store logs for all epochs
    best_f1 = 0.0
    for epoch in tqdm(range(1, num_epochs + 1), disable = training_verbose):
        if training_verbose:print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + "-"*16)

        model.train()
        running_loss = 0.0
        running_labels, running_preds = [], []
        
        for ecgs, labels in tqdm(train_loaders["train"]):
            ecgs, labels = ecgs.cuda(), labels.cuda()

            logits = model(ecgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*ecgs.size(0)
            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.50, 1, 0))
            running_labels.extend(labels), running_preds.extend(preds)

        if (scheduler is not None) and (not epoch > scheduler.T_max):
            scheduler.step()

        epoch_loss, epoch_f1 = running_loss/len(train_loaders["train"].dataset), f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        if training_verbose:
            print("{:<5} - loss:{:.4f}, f1:{:.4f}".format(
                "train", 
                epoch_loss, epoch_f1
            ))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_labels, running_preds = [], []
            for ecgs, labels in tqdm(train_loaders["val"], disable = not training_verbose):
                ecgs, labels = ecgs.cuda(), labels.cuda()

                logits = model(ecgs)
                loss = criterion(logits, labels)

                running_loss = running_loss + loss.item()*ecgs.size(0)
                labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.50, 1, 0))
                running_labels.extend(labels), running_preds.extend(preds)

        epoch_loss_val, epoch_classification_report_val = running_loss/len(train_loaders["val"].dataset), classification_report(
            running_labels, running_preds
        )
        epoch_f1_val = f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        if training_verbose:
            print("{:<5} - loss:{:.4f}, f1:{:.4f}".format(
                "val", 
                epoch_loss_val, epoch_f1_val
            ))
            print( 'validation report :', epoch_classification_report_val )

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1; torch.save(model.state_dict(), "{}/best.pth".format(save_ckp_dir))
    logs = {
        "epoch": epoch,
        "loss_train": epoch_loss,
        "loss_valid": epoch_loss_val,
        "F1_train": epoch_f1,
        "F1_train": epoch_f1_val,
        'validation report': epoch_classification_report_val
        }
    
    all_logs.append(logs)  # Append the logs for this epoch to the list
    
    # Save all_logs to the pickle file
    with open('my_train_logs.pkl', 'wb') as f:
        pickle.dump(all_logs, f)

    print("\nStart Evaluation ...\n" + " = "*16)
    model = torch.load("{}/best.pth".format(save_ckp_dir), map_location = "cuda")
    model = nn.DataParallel(model, device_ids = config["device_ids"])

    with torch.no_grad():
        model.eval()
        running_labels, running_preds = [], []
        for ecgs, labels in tqdm(train_loaders["val"], disable = not training_verbose):
            ecgs, labels = ecgs.cuda(), labels.cuda()

            logits = model(ecgs)

            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(torch.sigmoid(logits).detach().cpu().numpy())
            running_labels.extend(labels), running_preds.extend(preds)

    if config["is_multilabel"]:
        running_labels, running_preds = np.array(running_labels), np.array(running_preds)

        optimal_thresholds = thresholds_search(running_labels, running_preds)
        running_preds = np.stack([
            np.where(running_preds[:, cls] >= optimal_thresholds[cls], 1, 0) for cls in range(running_preds.shape[1])
        ]).transpose()
    val_loss, val_f1 = running_loss/len(train_loaders["val"].dataset), classification_report(
        running_labels, running_preds
    )
    print("{:<5} - loss:{:.4f}".format(
        "val", 
        val_loss
    ))
    print( 'validation report :', val_f1 )