from typing import Any

import hydra
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class LungColonCancerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_processed_dir: str,
        valid_processed_dir: str,
        test_processed_dir: str,
        augmentations: Any,
        valid_transforms: Any,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        batch_size: int = 32,
        subset_size: float | None = None,
    ):
        """
        Initialize the data module for the lung and colon cancer classification dataset.

        Parameters:
        train_processed_dir (str): Directory containing the training images.
        valid_processed_dir (str): Directory containing the validation images.
        test_processed_dir (str): Directory containing the test images.
        augmentations (Any): Augmentations to be applied to the training images.
        valid_transforms (Any): Transformations to be applied to the validation images.
        num_workers (int): Number of workers for the data loader. Defaults to 8.
        pin_memory (bool): Whether to pin the memory for the data loader. Defaults to True.
        persistent_workers (bool): Whether to use persistent workers for the data loader. Defaults to True.
        batch_size (int): Batch size for the data loader. Defaults to 32.
        subset_size (float | None): Fraction of the dataset to use for training, validation, and testing. Defaults to None.
        """
        super().__init__()
        self.train_data_dir = train_processed_dir
        self.valid_data_dir = valid_processed_dir
        self.test_data_dir = test_processed_dir
        self.augmentations = None
        self.valid_transforms = None
        self.subset_size = subset_size
        if augmentations:
            aug = hydra.utils.instantiate(augmentations)
            self.augmentations = v2.Compose(aug)
        if valid_transforms:
            transforms = hydra.utils.instantiate(valid_transforms)
            self.valid_transforms = v2.Compose(transforms)

        self.kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }

    def prepare_data(self):
        pass

    def subset_indices(self, dataset, subset_size) -> np.ndarray:
        train_ds_len = len(dataset)
        indices = np.arange(len(dataset))[: int(train_ds_len * self.subset_size)]
        return indices

    def setup(self, stage=None) -> None:
        # Set up the dataset for training and validation
        self.train_dataset = ImageFolder(root=self.train_data_dir, transform=self.augmentations)
        self.val_dataset = ImageFolder(root=self.valid_data_dir, transform=self.valid_transforms)
        self.test_dataset = ImageFolder(root=self.test_data_dir, transform=self.valid_transforms)

        if self.subset_size:
            print(f"Using subset of size {self.subset_size} for training, validation, and testing.")
            # Subset the dataset
            train_indices = self.subset_indices(self.train_dataset, self.subset_size)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)
            val_indices = self.subset_indices(self.val_dataset, self.subset_size)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices)
            test_indices = self.subset_indices(self.test_dataset, self.subset_size)
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            **self.kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            **self.kwargs,
        )
