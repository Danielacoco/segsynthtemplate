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


def _one_per_protocol(ds):
    """Return one sample per protocol using the first and last flat indices."""
    samples = {}
    indices = [0, len(ds) - 1]
    for idx in indices:
        sample = ds[idx]
        name = sample["protocol_name"]
        if name not in samples:
            samples[name] = sample
    return samples


@task_wrapper
def evaldata(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    num_protocols = len(datamodule.train_ds.protocol_registry)
    print(f"\nprotocol_registry : {datamodule.train_ds.protocol_registry}")
    print(f"num_channels      : {datamodule.train_ds.num_channels}")

    train_samples = _one_per_protocol(datamodule.train_ds)
    val_samples   = _one_per_protocol(datamodule.val_ds)
    test_samples  = _one_per_protocol(datamodule.test_ds)

    for protocol in datamodule.train_ds.protocol_registry:
        print(f"\n=== protocol: {protocol} ===")
        if protocol in train_samples:
            _print_sample("train", train_samples[protocol])
        if protocol in val_samples:
            _print_sample("val",   val_samples[protocol])
        if protocol in test_samples:
            _print_sample("test",  test_samples[protocol])

    # protocol_vec sanity checks across all splits and protocols
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for protocol, sample in samples.items():
            assert "protocol_vec" in sample, f"{split_name}/{protocol} sample missing protocol_vec"
            assert sample["protocol_vec"].shape == (num_protocols,), (
                f"{split_name}/{protocol} protocol_vec shape mismatch: "
                f"{sample['protocol_vec'].shape} vs ({num_protocols},)"
            )
            assert sample["protocol_vec"].sum() == 1.0, \
                f"{split_name}/{protocol} protocol_vec is not one-hot"
    print("\nprotocol_vec checks passed.")

    # visualise one sample per protocol per split
    protocols = list(datamodule.train_ds.protocol_registry.keys())
    n_rows = len(protocols) * 2  # image + label row per protocol
    n_cols = 3                   # axial, coronal, sagittal
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * len(protocols)))
        for p_idx, protocol in enumerate(protocols):
            if protocol not in samples:
                continue
            sample = samples[protocol]
            slicenum = sample["image"].shape[-1] // 2
            _plot_sample(axes[p_idx * 2], axes[p_idx * 2 + 1],
                         sample, f"{split_name} — {protocol}", slicenum)
        plt.tight_layout()
        fname = f"data_sample_multiprotocol_{split_name}.png"
        plt.savefig(fname)
        log.info(f"Saved visualisation to {fname}")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    evaldata(cfg)


if __name__ == "__main__":
    main()
