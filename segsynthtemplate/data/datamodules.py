import lightning as L
from torch.utils.data import DataLoader
from fetalsyngen.data.datasets import FetalTestDataset, FetalSynthDataset, FetalSynthGen, MultiProtocolDataset
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


class MultiProtocolDataModule(L.LightningDataModule):
    """DataModule for multi-protocol CoNeMos training.

    ``datasets`` is a list of dicts, one per source dataset. Each dict must
    have ``bids_path``, ``split_file``, and ``protocol_name`` keys. Multiple
    entries may share the same ``protocol_name`` (e.g. two FeTa cohorts).
    ``get_subjects()`` reads each split CSV and combines subjects across
    all datasets to produce train / val / test entries.

    Synthesis settings (generator, seed_path, apply_mri_augm) are passed
    separately via ``synth_configs``, a dict keyed by protocol name.
    ``train_type`` and ``val_type`` (``"synth"`` or ``"real"``) mirror the
    original ``DataModule`` — both use ``FetalSynthDataset`` internally.
    Test always uses ``FetalTestDataset`` (no generator).

    The split CSV for each dataset must have columns ``participant_id`` and
    ``splits``, where ``splits`` values match ``train_split``, ``val_split``,
    and ``test_split`` (e.g. ``"train"``, ``"val"``, ``"test_i"``).

    Args:
        datasets: List of dicts, each with ``bids_path``, ``split_file``,
            and ``protocol_name``. Multiple entries may share a protocol name.
        label_map_csv: CSV with columns ``protocol``, ``raw_label``,
            ``channel``. Maps raw per-protocol label integers to shared
            output channel indices.
        train_split: Value in the ``splits`` column for training subjects.
        val_split: Value in the ``splits`` column for validation subjects.
        test_split: Value in the ``splits`` column for test subjects.
        transforms: MONAI transforms applied to val and test samples.
        synth_configs: Optional dict ``{protocol_name: {...}}`` for datasets
            that use synthesis during training. Each inner dict may contain:
            ``train_type`` (``"synth"``, default ``"real"``), ``generator``
            (required if synth), ``seed_path``, ``apply_mri_augm``.
        num_workers: DataLoader workers.
        batch_size: Batch size for all dataloaders.
        img_suffix: Image file suffix.
        seg_suffix: Segmentation file suffix.
    """

    def __init__(
        self,
        datasets: list[dict],
        label_map_csv: str,
        train_split: str,
        val_split: str,
        test_split: str,
        transforms: monai.transforms.Compose,
        train_type: str = "real",
        val_type: str = "real",
        synth_configs: dict[str, dict] | None = None,
        num_workers: int = 1,
        batch_size: int = 1,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
    ):
        super().__init__()
        self.datasets = datasets
        self.label_map_csv = label_map_csv
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.transforms = transforms
        self.train_type = train_type
        self.val_type = val_type
        self.synth_configs = synth_configs or {}
        assert self.train_type in ("synth", "real")
        assert self.val_type in ("synth", "real")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix

        train_entries, val_entries, test_entries = self.get_subjects()

        self.train_ds = MultiProtocolDataset(
            dataset_entries=train_entries,
            label_map_csv=self.label_map_csv,
            img_suffix=self.img_suffix,
            seg_suffix=self.seg_suffix,
        )
        self.val_ds = MultiProtocolDataset(
            dataset_entries=val_entries,
            label_map_csv=self.label_map_csv,
            img_suffix=self.img_suffix,
            seg_suffix=self.seg_suffix,
            transforms=self.transforms,
        )
        self.test_ds = MultiProtocolDataset(
            dataset_entries=test_entries,
            label_map_csv=self.label_map_csv,
            img_suffix=self.img_suffix,
            seg_suffix=self.seg_suffix,
            transforms=self.transforms,
        )

        logging.info(f"Train dataset size: {len(self.train_ds)}")
        logging.info(f"Val dataset size: {len(self.val_ds)}")
        logging.info(f"Test dataset size: {len(self.test_ds)}")

    def get_subjects(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Read each dataset's split CSV and return three lists of entries
        (train, val, test) with ``sub_list`` pre-filled.

        Loops over ``self.datasets`` exactly like the original
        ``DataModule.get_subjects()``, but across multiple datasets.
        """
        train_entries, val_entries, test_entries = [], [], []

        for ds in self.datasets:
            bids_path     = ds["bids_path"]
            split_file    = ds["split_file"]
            protocol_name = ds["protocol_name"]

            split_df = pd.read_csv(split_file)
            assert "participant_id" in split_df.columns, (
                f"split_file '{split_file}' is missing 'participant_id' column"
            )
            assert "splits" in split_df.columns, (
                f"split_file '{split_file}' is missing 'splits' column"
            )

            base = {"bids_path": bids_path, "protocol_name": protocol_name}
            synth = self.synth_configs.get(protocol_name, {})

            train_entries.append({
                **base,
                "sub_list":   split_df[split_df.splits == self.train_split].participant_id.tolist(),
                "train_type": self.train_type,
                "is_test":    False,
                # forward synth-only keys when present
                **{k: synth[k] for k in ("generator", "seed_path", "apply_mri_augm") if k in synth},
            })
            val_entries.append({
                **base,
                "sub_list":   split_df[split_df.splits == self.val_split].participant_id.tolist(),
                "train_type": self.val_type,
                "is_test":    False,
                # forward synth-only keys for val if val_type="synth"
                **({k: synth[k] for k in ("generator", "seed_path") if k in synth}
                   if self.val_type == "synth" else {}),
            })
            test_entries.append({
                **base,
                "sub_list":   split_df[split_df.splits == self.test_split].participant_id.tolist(),
                "is_test":    True,   # always FetalTestDataset, train_type ignored
            })

        return train_entries, val_entries, test_entries

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
