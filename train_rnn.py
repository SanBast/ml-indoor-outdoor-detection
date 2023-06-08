import torch
import pytorch_lightning as pl
from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torchmetrics import Accuracy, F1Score 

from rnn_model import IndoorPredictor
from config import TRAIN_PARAMS, EARLY_STOPPING_PARAMS
from data_module import IndoorDataModule
from preprocess import DataframeToSeq

def train(model, data_module):
    checkpoint_callback = ModelCheckpoint(
        dirpath=TRAIN_PARAMS['ckpt_dir'],
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(**EARLY_STOPPING_PARAMS)

    logger = TensorBoardLogger("/lightning_logs", name="indoor")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=TRAIN_PARAMS['n_epochs'],
        accelerator=TRAIN_PARAMS['accelerator'],
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)


def main():

    df2seq = DataframeToSeq()
    train_sequences, val_sequences = df2seq.sequences()

    model = IndoorPredictor(
        type_rnn='lstm',
        n_features=len(train_sequences[0][0].columns),
        n_classes=1
    )

    data_module = IndoorDataModule(
        train_sequences=train_sequences, 
        test_sequences=[], 
        val_sequences=val_sequences, 
        batch_size=TRAIN_PARAMS['batch_size']
        )

    assert torch.cuda.is_available()

    train(model, data_module)


if __name__ == '__main__':
    main()