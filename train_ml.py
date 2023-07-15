from multiprocessing import cpu_count

import torch

import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from config import DATA_CONFIG
from preprocess import DataframeToSeq
from feature_extraction import FeatExtractor, PCAExtractor


def main():
    df2seq = DataframeToSeq(**DATA_CONFIG)
    train_df, y_train = df2seq.return_nested_df()
    
    feat_extract = FeatExtractor(type="all")
    X_train = feat_extract.preprocess(train_df)
    
    # recommended
    pca_extract = PCAExtractor(n_components="auto")
    X_pca = pca_extract.preprocess(X_train)
    
    model = RandomForestClassifier(
            max_features=0.5,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
    
    model.fit(X_pca, y_train)
    filename = 'last_RF.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    main()