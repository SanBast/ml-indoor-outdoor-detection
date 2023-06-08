import pandas as pd
import numpy as np
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class IndoorDataset(Dataset):
  def __init__(self, sequences):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self,idx):
    sequence, label = self.sequences[idx]
    return dict(
        sequence=torch.tensor(sequence.to_numpy(), dtype=torch.float),
        label=torch.tensor(label, dtype=torch.float)
    )


class IndoorDataModule(pl.LightningDataModule):
  def __init__(self, train_sequences, test_sequences, val_sequences, batch_size):
    super().__init__()
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.val_sequences = val_sequences
    self.batch_size = batch_size
  
  def setup(self, stage=None):
    self.train_dataset = IndoorDataset(self.train_sequences)
    self.val_dataset = IndoorDataset(self.val_sequences)
    self.test_dataset = IndoorDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cpu_count()
    )
  
  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cpu_count()
    )
  
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=cpu_count()
    )