from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from channel_attention.utils.load_bcic import load_bcic


class BCICIV(pl.LightningDataModule):
    all_subject_ids = list(range(1, 10))
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BCICIV, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])

    @staticmethod
    def _z_scale(X, X_test):
        for ch_idx in range(X.shape[1]):
            sc = StandardScaler()
            X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
            X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
        return X, X_test

    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))


class BCICIV2a(BCICIV):
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]

    def __init__(self, preprocessing_dict, subject_id):
        super(BCICIV2a, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic(subject_id=self.subject_id, dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BCICIV._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BCICIV._make_tensor_dataset(X, y)
        self.test_dataset = BCICIV._make_tensor_dataset(X_test, y_test)


class BCICIV2b(BCICIV):
    class_names = ["hand(L)", "hand(R)"]

    def __init__(self, preprocessing_dict, subject_id):
        super(BCICIV2b, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic(subject_id=self.subject_id, dataset="2b",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("session")
        train_datasets = [splitted_ds[f"session_{session}"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"session_{session}"] for session in [3, 4]]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for train_dataset in train_datasets for run
             in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for test_dataset in test_datasets for run in
             test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BCICIV._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BCICIV._make_tensor_dataset(X, y)
        self.test_dataset = BCICIV._make_tensor_dataset(X_test, y_test)
