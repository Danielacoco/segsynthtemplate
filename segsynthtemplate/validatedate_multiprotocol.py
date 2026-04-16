"""Sanity-check script for MultiProtocolDataModule and MultiProtocolDataset.

Mirrors validatedate.py but adds checks specific to the multi-protocol setup:
- protocol_vec shape and values
- label remapping (only valid channel indices present)
- one sample per split visualised in 3 orientations

Run with:
    python segsynthtemplate/validatedate_multiprotocol.py \
        experiment=mthesis/validate_multiprotocol
"""

from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import rootutils
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import DictConfig
from segsynthtemplate.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _print_sample(name: str, sample: dict) -> None:
    img, lbl = sample["image"], sample["label"]
    print(f"\n[{name}]")
    print(f"  image : shape={tuple(img.shape)}  dtype={img.dtype}  "
          f"min={img.min():.3f}  max={img.max():.3f}")
    print(f"  label : shape={tuple(lbl.shape)}  dtype={lbl.dtype}  "
          f"unique={sorted(lbl.unique().tolist())}")
    if "protocol_vec" in sample:
        print(f"  protocol_name : {sample['protocol_name']}")
        print(f"  protocol_vec  : {sample['protocol_vec'].tolist()}")
    print(f"  name  : {sample['name']}")


def _plot_sample(ax_row_img, ax_row_lbl, sample: dict, title: str, slicenum: int) -> None:
    img = sample["image"][0]
    lbl = sample["label"][0]
    for col, (img_sl, lbl_sl) in enumerate([
        (img[slicenum, :, :], lbl[slicenum, :, :]),  # axial
        (img[:, slicenum, :], lbl[:, slicenum, :]),  # coronal
        (img[:, :, slicenum], lbl[:, :, slicenum]),  # sagittal
    ]):
        ax_row_img[col].imshow(img_sl.cpu(), cmap="gray")
        ax_row_img[col].set_title(f"{title} image")
        ax_row_img[col].axis("off")
        ax_row_lbl[col].imshow(lbl_sl.cpu(), cmap="jet")
        ax_row_lbl[col].set_title(f"{title} label")
        ax_row_lbl[col].axis("off")


@task_wrapper
def evaldata(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    train_sample = datamodule.train_ds[0]
    val_sample   = datamodule.val_ds[0]
    test_sample  = datamodule.test_ds[0]

    _print_sample("train", train_sample)
    _print_sample("val",   val_sample)
    _print_sample("test",  test_sample)

    # protocol_vec sanity checks (train and val carry it; test does too)
    num_protocols = len(datamodule.train_ds.protocol_registry)
    print(f"\nprotocol_registry : {datamodule.train_ds.protocol_registry}")
    print(f"num_channels      : {datamodule.train_ds.num_channels}")
    for split_name, sample in [("train", train_sample), ("val", val_sample), ("test", test_sample)]:
        assert "protocol_vec" in sample, f"{split_name} sample missing protocol_vec"
        assert sample["protocol_vec"].shape == (num_protocols,), (
            f"{split_name} protocol_vec shape mismatch: "
            f"{sample['protocol_vec'].shape} vs ({num_protocols},)"
        )
        assert sample["protocol_vec"].sum() == 1.0, \
            f"{split_name} protocol_vec is not one-hot"
    print("\nprotocol_vec checks passed.")

    # visualise one sample per split in 3 orientations
    slicenum = train_sample["image"].shape[-1] // 2
    fig, axes = plt.subplots(6, 3, figsize=(12, 18))
    _plot_sample(axes[0], axes[1], train_sample, "Train",      slicenum)
    _plot_sample(axes[2], axes[3], val_sample,   "Validation", slicenum)
    _plot_sample(axes[4], axes[5], test_sample,  "Test",       slicenum)
    plt.tight_layout()
    plt.savefig("data_sample_multiprotocol.png")
    log.info("Saved visualisation to data_sample_multiprotocol.png")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    evaldata(cfg)


if __name__ == "__main__":
    main()
