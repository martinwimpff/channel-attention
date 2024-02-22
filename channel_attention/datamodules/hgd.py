from typing import Optional

from braindecode.datasets.base import BaseConcatDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from channel_attention.utils.load_hgd import load_hgd


class HighGamma(BaseDataModule):
    all_subject_ids = list(range(1, 15))
    class_names = ["feet", "hand(L)", "rest", "hand(R)"]

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_hgd(subject_ids=[self.subject_id],
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


class HighGammaLOSO(HighGamma):
    val_dataset = None

    def __init__(self, preprocessing_dict, subject_id):
        super(HighGammaLOSO, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        # download the dataset
        self.dataset = load_hgd(subject_ids=self.all_subject_ids,
                                           preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        train_subjects = [subj for subj in self.all_subject_ids if subj != self.subject_id]
        train_dataset = BaseConcatDataset([self.dataset.split("run")["train"].split("subject")[str(subj)] for subj in train_subjects])
        val_dataset = BaseConcatDataset([self.dataset.split("run")["test"].split("subject")[str(subj)] for subj in train_subjects])
        test_dataset = self.dataset.split("subject")[str(self.subject_id)].split("run")["test"]

        # load the test data
        X_test = test_dataset.datasets[0].windows.load_data()._data
        y_test = np.array(test_dataset.datasets[0].y)

        if self.preprocessing_dict["z_scale"]:
            for ch_idx in range(X_test.shape[1]):
                sc = StandardScaler()
                X_test[:, ch_idx, :] = sc.fit_transform(X_test[:, ch_idx, :])

        # make datasets
        self.train_dataset = CustomDataset(train_dataset)
        self.val_dataset = CustomDataset(val_dataset)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True, collate_fn=z_scale_collate_fn if self.preprocessing_dict.get("z_scale", False) else None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.val_dataset, batch_size=self.preprocessing_dict["batch_size"],
                collate_fn=z_scale_collate_fn if self.preprocessing_dict.get("z_scale", False) else None)


# Note: this is a datset that supports cont. data loading
# (w/o loading the whole dataset in the memory at once)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = [idx < _ for _ in self.dataset.cumulative_sizes].index(True)
        idx = idx - self.dataset.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else idx
        X = self.dataset.datasets[dataset_idx].windows.load_data()._data[idx]
        y = self.dataset.datasets[dataset_idx].y[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y).type(torch.LongTensor)


def z_scale_collate_fn(batch):
    data, labels = zip(*batch)
    x = torch.stack(data)
    x_scaled = (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=True, keepdim=True)
    return x_scaled, torch.stack(labels)
