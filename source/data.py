
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
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import captum.attr as attr
import matplotlib.pyplot as pyplot
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_path, 
        config, 
        augment = False, 
    ):
        self.df_path, self.data_path,  = df_path, data_path, 
        self.df = pd.read_csv(self.df_path)

        self.config = config
        self.augment = augment

    def __len__(self, 
    ):
        return len(self.df)

    def drop_lead(self, 
        ecg, 
    ):
        if random.random() >= 0.5:
            ecg[np.random.randint(len(self.config["ecg_leads"])), :] = 0.0
        return ecg

    def __getitem__(self, 
        index, 
    ):
        row = self.df.iloc[index]

        ecg = np.load("{}/{}.npy".format(self.data_path, row["id"]))[self.config["ecg_leads"], :]
        ecg = pad_sequences(ecg, self.config["ecg_length"], "float64", 
            "post", "post", 
        )
        if self.augment:
            ecg = self.drop_lead(ecg)
        ecg = torch.tensor(ecg).float()

        if not self.config["is_multilabel"]:
            label = row["label"]
        else:
            label = row[[col for col in list(row.index) if "label_" in col]].values.astype("float64")

        return ecg, label