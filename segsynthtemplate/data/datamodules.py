import lightning as L
from torch.utils.data import DataLoader
from fetalsyngen.data.datasets import FetalTestDataset, FetalSynthDataset, FetalSynthGen
import pandas as pd
import monai
import logging


class DataModule(L.LightningDataModule):

    def __init__(
        self,
        split_file: str,
        bids_path: str,
        seed_path: str,
        train_type: str,
        train_split: str,
        val_type: str,
        val_split: str,
        test_split: str,
        generator: FetalSynthGen | None,
        transforms: monai.transforms.Compose,
        num_workers: int = 1,
        batch_size: int = 1,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        apply_mri_augm: bool = False,
        load_segmentations: bool = True,
    ):
        super().__init__()
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.split_file = split_file
        self.bids_path = bids_path
        self.seed_path = seed_path
        self.train_type = train_type
        self.train_split = train_split
        self.apply_mri_augm = apply_mri_augm
        self.val_type = val_type
        self.val_split = val_split
        self.test_split = test_split
        self.generator = generator
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transform = transforms
        self.load_segmentations = load_segmentations
        assert self.train_type in ["synth", "real"]
        assert self.val_type in ["synth", "real", "test"]

        self.train_subjects, self.val_subjects, self.test_subjects = self.get_subjects()
        # Initialize datasets
        if self.train_type == "synth":

            assert self.generator is not None

            self.train_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.train_subjects,
                load_image=False,
                image_as_intensity=False,
                generator=self.generator,
                img_suffix=self.img_suffix,
                seg_suffix=self.seg_suffix,
                apply_mri_augm=self.apply_mri_augm,
            )
        elif self.train_type == "real":
            self.train_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.train_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator,
                img_suffix=self.img_suffix,
                seg_suffix=self.seg_suffix,
                apply_mri_augm=self.apply_mri_augm,
            )

        if self.val_type == "synth":
            assert self.generator is not None
            self.val_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.val_subjects,
                load_image=False,
                image_as_intensity=False,
                generator=self.generator,
                img_suffix=self.img_suffix,
                seg_suffix=self.seg_suffix,
            )
        elif self.val_type == "real":
            self.val_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.val_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator,
                img_suffix=self.img_suffix,
                seg_suffix=self.seg_suffix,
            )
        elif self.val_type == "test":
            self.val_ds = FetalTestDataset(
                bids_path=self.bids_path,
                sub_list=self.val_subjects,
                transforms=self.transform,
                img_suffix=self.img_suffix,
                seg_suffix=self.seg_suffix,
            )

        self.test_ds = FetalTestDataset(
            bids_path=self.bids_path,
            sub_list=self.test_subjects,
            transforms=self.transform,
            img_suffix=self.img_suffix,
            seg_suffix=self.seg_suffix,
            load_segmentations=self.load_segmentations,
        )
        # log dataset size
        logging.info(f"Train dataset size: {len(self.train_ds)}")
        logging.info(f"Val dataset size: {len(self.val_ds)}")
        logging.info(f"Test dataset size: {len(self.test_ds)}")

    def get_subjects(self):
        split_df = pd.read_csv(self.split_file)
        assert "participant_id" in split_df.columns
        assert "splits" in split_df.columns

        train_subjects = split_df[
            split_df.splits == self.train_split
        ].participant_id.tolist()

        val_subjects = split_df[
            split_df.splits == self.val_split
        ].participant_id.tolist()

        test_subjects = split_df[
            split_df.splits == self.test_split
        ].participant_id.tolist()

        return train_subjects, val_subjects, test_subjects

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
        )
