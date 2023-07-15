import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

from multiprocessing import cpu_count

import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")


class RNNClassifier(nn.Module):
  '''
  RNN architecture. It allows you to choose between LSTM and GRU. LSTM is recommended. 
  '''
  def __init__(self, type_rnn, n_features, n_classes, n_hidden=256, n_layers=3):
    super().__init__()
    self.type_rnn = type_rnn

    if self.type_rnn=='lstm':
      self.rnn = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          batch_first=True,
          dropout=0
      )
    elif self.type_rnn=='gru':
      self.rnn = nn.GRU(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          batch_first=True,
          dropout=0
      )
    self.classifier = nn.Linear(n_hidden, n_classes)

    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):

    self.rnn.flatten_parameters()
    if self.type_rnn=='lstm':
      _, (hidden, _) = self.rnn(x)
    elif self.type_rnn=='gru':
      _, hidden = self.rnn(x)

    out = hidden[-1]
    out = self.classifier(out)
    out = self.sigmoid(out)

    return torch.squeeze(out,1)


class BidirLSTM(nn.Module):
    def __init__(self, in_dim, num_classes, bidirectional, activation, n_hidden=256, num_layers=3):
        super(BidirLSTM, self).__init__()

        self.arch = 'lstm'
        self.n_hidden = n_hidden
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        avail_activations = {
          'relu': nn.ReLU(),
          'gelu': nn.GELU(),
          'elu': nn.ELU()
        }

        self.activation = avail_activations[activation]

        self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=n_hidden,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional
            )

        self.hidden2label = nn.Sequential(
            nn.Linear(n_hidden*self.num_dir, n_hidden),
            self.activation,
            nn.Dropout(),
            nn.Linear(n_hidden, n_hidden),
            self.activation,
            nn.Dropout(),
            nn.Linear(n_hidden, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, batch, self.n_hidden).cuda())
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, batch, self.n_hidden).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, batch, self.n_hidden))
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, batch, self.n_hidden))
        return (h0, c0)

    def forward(self, x): 
      # Note: x is (batch_size, T, F), permute to (T, batch_size, F)
        x = x.permute(1, 0, 2)

        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, (h, c) = self.lstm(x, self.init_hidden(x.shape[1]))
        y  = self.hidden2label(lstm_out[-1])
        out = self.sigmoid(y)

        return torch.squeeze(out,1)


class IndoorPredictor(pl.LightningModule):
  '''
  Pytorch-lightning requires this as Class to inject data to the model
  '''
  def __init__(self, type_rnn: str, n_features: int, n_classes: int):
    super().__init__()
    '''
  BidirLSTM deprecated -- not to be used

    self.model = BidirLSTM(
        in_dim=n_features,
        num_classes=n_classes,
        bidirectional=False
    )
    '''
    self.model = RNNClassifier(
        type_rnn=type_rnn,
        n_features=n_features, 
        n_classes=n_classes
    )
    self.criterion = nn.BCELoss()
  

  def forward(self, x, labels=None):
    output = self.model(x).cuda()
    loss = 0
    if labels is not None:
      loss = self.criterion(output, labels)
    return loss, output
  

  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    threshold = torch.tensor([0.5]).cuda()
    predictions = (outputs>threshold).float()*1
    correct_results_sum = (predictions == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    step_accuracy = acc

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    threshold = torch.tensor([0.5]).cuda()
    predictions = (outputs>threshold).float()*1
    correct_results_sum = (predictions == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    step_accuracy = acc
    step_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
    self.log("val_f1score", step_f1, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy, "f1-score": step_f1}
  
  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    threshold = torch.tensor([0.5]).cuda()
    predictions = (outputs>threshold).float()*1
    correct_results_sum = (predictions == labels).sum().float()
    acc = correct_results_sum/labels.shape[0]
    step_accuracy = acc

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.0001)


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
  '''
  Dividing various sequences in train, test, val. 
  Required module for Pytorch Lightning training
  '''
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
