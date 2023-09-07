from typing import Optional

import numpy as np

from .base import BaseDataModule
from channel_attention.utils.load_hgd import load_hgd


class HighGamma(BaseDataModule):
    all_subject_ids = list(range(1, 15))
    class_names = ["feet", "hand(L)", "rest", "hand(R)"]

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_hgd(subject_id=self.subject_id,
                                preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("run")
        train_dataset, test_dataset = splitted_ds["train"], splitted_ds["test"]

        # load the data
        X = train_dataset.datasets[0].windows.load_data()._data
        y = np.array(train_dataset.datasets[0].y)
        X_test = test_dataset.datasets[0].windows.load_data()._data
        y_test = np.array(test_dataset.datasets[0].y)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
