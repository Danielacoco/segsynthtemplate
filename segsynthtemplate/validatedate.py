from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from segsynthtemplate.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaldata(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.get("ckpt_path"):
        log.info(f"Loading model weights from <{cfg.ckpt_path}>")
        state_dict = torch.load(cfg.ckpt_path)["state_dict"]
        # remove the last layer of the model
        state_dict = {
            k: v for k, v in state_dict.items() if "net.model.2.0.conv" not in k
        }
        model.load_state_dict(state_dict, strict=False)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    print("Train dataset")
    train_sample = datamodule.train_ds[0]
    valid_sample = datamodule.val_ds[0]
    test_sample = datamodule.test_ds[0]
    print(
        f'IMAGE: Type: {type(train_sample["image"])} Dtype: {train_sample["image"].dtype} Shape: {train_sample["image"].shape} Min: {train_sample["image"].min()} Max: {train_sample["image"].max()}'
    )
    print(
        f"LABEL TYPE: {type(train_sample['label'])} Dtype: {train_sample['label'].dtype} SHAPE: {train_sample['label'].shape} Min: {train_sample['label'].min()} Max: {train_sample['label'].max()}"
    )

    print("Validation dataset")
    print(
        f"IMAGE Type: {type(valid_sample['image'])} Shape: {valid_sample['image'].shape}  Dtype: {valid_sample['image'].dtype} Min: {valid_sample['image'].min()} Max: {valid_sample['image'].max()}"
    )
    print(
        f"LABEL Type: {type(valid_sample['label'])} Shape: {valid_sample['label'].shape}  Dtype: {valid_sample['label'].dtype} Min: {valid_sample['label'].min()} Max: {valid_sample['label'].max()}"
    )

    print("Test dataset")
    print(
        f'IMAGE: Type: {type(test_sample["image"])} Dtype: {test_sample["image"].dtype} Shape: {test_sample["image"].shape} Min: {test_sample["image"].min()} Max: {test_sample["image"].max()}'
    )
    print(
        f"LABEL TYPE: {type(test_sample['label'])} Dtype: {test_sample['label'].dtype} SHAPE: {test_sample['label'].shape} Min: {test_sample['label'].min()} Max: {test_sample['label'].max()}"
    )
    slicenum = 128
    # plot 3 images and their labels side by side in 3 orientations
    fig, ax = plt.subplots(6, 3, figsize=(10, 15))
    # image ax
    # segm ax
    ax[0, 0].imshow(train_sample["image"][0, slicenum, :, :], cmap="gray")
    ax[0, 0].set_title("Train Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(valid_sample["image"][0, slicenum, :, :], cmap="gray")
    ax[0, 1].set_title("Validation Image")
    ax[0, 1].axis("off")
    ax[0, 2].imshow(test_sample["image"][0, slicenum, :, :], cmap="gray")
    ax[0, 2].set_title("Test Image")
    ax[0, 2].axis("off")
    ax[1, 0].imshow(train_sample["label"][0, slicenum, :, :], cmap="jet")
    ax[1, 0].set_title("Train Label")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(valid_sample["label"][0, slicenum, :, :], cmap="jet")
    ax[1, 1].set_title("Validation Label")
    ax[1, 1].axis("off")
    ax[1, 2].imshow(test_sample["label"][0, slicenum, :, :], cmap="jet")
    ax[1, 2].set_title("Test Label")
    ax[1, 2].axis("off")
    # image cor
    # segm cor
    ax[2, 0].imshow(train_sample["image"][0, :, slicenum, :], cmap="gray")
    ax[2, 0].set_title("Train Image")
    ax[2, 0].axis("off")
    ax[2, 1].imshow(valid_sample["image"][0, :, slicenum, :], cmap="gray")
    ax[2, 1].set_title("Validation Image")
    ax[2, 1].axis("off")
    ax[2, 2].imshow(test_sample["image"][0, :, slicenum, :], cmap="gray")
    ax[2, 2].set_title("Test Image")
    ax[2, 2].axis("off")
    ax[3, 0].imshow(train_sample["label"][0, :, slicenum, :], cmap="jet")
    ax[3, 0].set_title("Train Label")
    ax[3, 0].axis("off")
    ax[3, 1].imshow(valid_sample["label"][0, :, slicenum, :], cmap="jet")
    ax[3, 1].set_title("Validation Label")
    ax[3, 1].axis("off")
    ax[3, 2].imshow(test_sample["label"][0, :, slicenum, :], cmap="jet")
    ax[3, 2].set_title("Test Label")
    ax[3, 2].axis("off")
    # image sag
    # segm sag
    ax[4, 0].imshow(train_sample["image"][0, :, :, slicenum], cmap="gray")
    ax[4, 0].set_title("Train Image")
    ax[4, 0].axis("off")
    ax[4, 1].imshow(valid_sample["image"][0, :, :, slicenum], cmap="gray")
    ax[4, 1].set_title("Validation Image")
    ax[4, 1].axis("off")
    ax[4, 2].imshow(test_sample["image"][0, :, :, slicenum], cmap="gray")
    ax[4, 2].set_title("Test Image")
    ax[4, 2].axis("off")
    ax[5, 0].imshow(train_sample["label"][0, :, :, slicenum], cmap="jet")
    ax[5, 0].set_title("Train Label")
    ax[5, 0].axis("off")
    ax[5, 1].imshow(valid_sample["label"][0, :, :, slicenum], cmap="jet")
    ax[5, 1].set_title("Validation Label")
    ax[5, 1].axis("off")
    ax[5, 2].imshow(test_sample["label"][0, :, :, slicenum], cmap="jet")
    ax[5, 2].set_title("Test Label")
    ax[5, 2].axis("off")

    plt.tight_layout()
    # display the plot
    plt.savefig("data_sample.png")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    evaldata(cfg)


if __name__ == "__main__":
    main()
