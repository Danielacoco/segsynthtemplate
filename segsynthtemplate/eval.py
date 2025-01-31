from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import monai.data
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import warnings
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from monai.metrics import DiceMetric, compute_hausdorff_distance

# one hot encode the labels
from monai.transforms import AsDiscrete
import pandas as pd

# suppress warnings
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
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
    )

    dice_metric = DiceMetric(include_background=False, return_with_label=True)
    onehoteencoder = AsDiscrete(to_onehot=cfg.model.net.out_channels)
    with torch.no_grad():
        for ckpt_path in cfg.get("ckpt_paths", []):
            exp_name = ckpt_path.split("/")[-5] + "/" + ckpt_path.split("/")[-3]
            log.info(f"Loading model weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            # move model to device
            model.to(cfg.device)
            log.info(f"Saving predictions to {cfg.get('output_dir')}")
            for test_split in cfg.get("test_splits"):

                log.info(f"Testing {exp_name} on split {test_split}")
                out_pred = Path(cfg["save_path"]) / f"{exp_name}/{test_split}"
                out_pred.mkdir(parents=True, exist_ok=True)

                cfg.data.test_split = test_split
                log.info(
                    f"Instantiating datamodule <{cfg.data._target_}> with test split {test_split}"
                )
                datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
                test_ds = datamodule.test_dataloader().dataset
                testsplit_df = []
                for tidx in tqdm(range(len(test_ds))):
                    test_data = test_ds[tidx]
                    image = test_data["image"]
                    label = test_data["label"]
                    name = test_data["name"]

                    # pred
                    pred = model.predict(image.unsqueeze(0).to(cfg.device))

                    pred_1h = onehoteencoder(pred.unsqueeze(0))
                    gt_1h = onehoteencoder(label.unsqueeze(0).to(cfg.device))

                    # calculate dice
                    dice = (
                        dice_metric(
                            pred_1h,
                            gt_1h,
                        )
                        .cpu()
                        .numpy()
                    )
                    hd95th = (
                        (
                            compute_hausdorff_distance(
                                pred_1h,
                                gt_1h,
                                percentile=95,
                                include_background=False,
                                spacing=cfg.data.generator.resolution,
                            )
                            .cpu()
                            .numpy()
                        )
                        if "hd95th" in cfg.metrics
                        else None
                    )
                    mean_dice = dice[1:].mean()

                    pred = monai.data.meta_tensor.MetaTensor(pred).copy_meta_from(label)
                    pred_data = {"label": pred, "image": image}
                    pred = test_ds.reverse_transform(pred_data)
                    # print pred meta dict
                    pred_data["label"].meta["name"] = name
                    output_dir = out_pred / f"{name}/anat/"
                    output_dir.mkdir(exist_ok=True, parents=True)
                    nib_image = nib.Nifti1Image(
                        pred_data["label"][0].cpu().numpy().astype("int8"),
                        affine=pred_data["label"].meta["affine"],
                    )
                    nib.save(
                        nib_image,
                        output_dir / f"{name}_dcs-{mean_dice:.3f}_pred.nii.gz",
                    )
                    subj_res = [
                        {
                            "Metric": "DSC",
                            "Value": dice[x][0],
                            "subj": name,
                            "Label": x,
                            "Split": test_split,
                            "Exp": exp_name,
                        }
                        for x in range(1, len(dice))
                    ]
                    subj_res_hd = (
                        [
                            {
                                "Metric": "HD95",
                                "Value": hd95th[x][0],
                                "subj": name,
                                "Label": x,
                                "Split": test_split,
                                "Exp": exp_name,
                            }
                            for x in range(1, len(hd95th))
                        ]
                        if "hd95th" in cfg.metrics
                        else []
                    )
                    testsplit_df.extend(subj_res + subj_res_hd)

                testsplit_df = pd.DataFrame(testsplit_df)

                testsplit_df.to_csv(
                    output_dir.parent.parent / "metrics.csv", index=False
                )
                experiment_mean_dsc = testsplit_df[testsplit_df["Metric"] == "DSC"][
                    "Value"
                ].mean()
                print(
                    f"Experiment {exp_name} split {test_split} done | DSC: {experiment_mean_dsc:.3f}"
                )

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
