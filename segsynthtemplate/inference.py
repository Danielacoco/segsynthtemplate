from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import monai.data
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
import warnings
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import numpy as np

# one hot encode the labels

# suppress warnings
warnings.filterwarnings("ignore")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from segsynthtemplate.utils import (
    RankedLogger,
    extras,
    get_metric_value,
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
            output_dir = Path(cfg["save_path"])
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

                for tidx in tqdm(range(len(test_ds))):
                    test_data = test_ds[tidx]
                    image = test_data["image"]
                    name = test_data["name"]
                    # pred
                    pred = model.predict(image.unsqueeze(0).to(cfg.device))

                    pred = monai.data.meta_tensor.MetaTensor(pred).copy_meta_from(image)
                    pred_data = {"label": pred, "image": image}
                    pred_orgi_space = test_ds.reverse_transform(pred_data)
                    # print pred meta dict
                    pred_orgi_space["label"].meta["name"] = name
                    output_dir = (
                        out_pred / f"{name}/anat/"
                        if "ses" not in name
                        else out_pred
                        / f"{name.split('_')[0]}/{name.split('_')[1]}/anat/"
                    )
                    output_dir.mkdir(exist_ok=True, parents=True)
                    nib_image = nib.Nifti1Image(
                        pred_orgi_space["label"][0].cpu().numpy().astype("int8"),
                        affine=pred_orgi_space["image"].meta["affine"],
                    )
                    modelname = "inference"
                    nib.save(
                        nib_image,
                        output_dir / f"{name}_seg-{modelname}_pred.nii.gz",
                    )

    return {}, {}


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="inference_lisa.yaml"
)
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
