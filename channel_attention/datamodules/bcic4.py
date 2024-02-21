from typing import Optional

import numpy as np

from .base import BaseDataModule
from channel_attention.utils.load_bcic4 import load_bcic4


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]

    def __init__(self, preprocessing_dict, subject_id):
        super(BCICIV2a, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_id=self.subject_id, dataset="2a",
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
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
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
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
