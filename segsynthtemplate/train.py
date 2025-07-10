from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from typing import Sequence
from torch.nn import Module

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

    if cfg.get("ckpt_path"):
        log.info(f"Loading model weights (partial) from <{cfg.ckpt_path}>")
        load_checkpoint_weights(
            model=model,
            ckpt_path=cfg.ckpt_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            # anything you were already removing, e.g. your final conv layer:
            # ignore_layers=["net.model.2.0.conv"],
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            # note: we no longer need to pass `ckpt_path` here,
            # since weights are already in the model
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )
    train_metrics = trainer.callback_metrics
    print(train_metrics)
    return train_metrics


def load_checkpoint_weights(
    model: Module,
    ckpt_path: str,
    ignore_layers: Sequence[str] = (),
    map_location: str = "",
) -> None:
    """
    Load matching weights from a checkpoint into `model`, skipping any layers
    whose names contain one of the `ignore_layers` substrings, missing keys,
    or shape mismatches. Prints a warning for each skipped key.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("state_dict", ckpt)
    model_dict = model.state_dict()

    loaded_keys = []
    skipped = []

    for name, param in state_dict.items():
        if any(ign in name for ign in ignore_layers):
            skipped.append((name, f"ignored by pattern {ignore_layers}"))
            continue

        if name not in model_dict:
            skipped.append((name, "not found in model"))
            continue

        if param.shape != model_dict[name].shape:
            skipped.append(
                (
                    name,
                    f"shape mismatch checkpoint {tuple(param.shape)} vs model {tuple(model_dict[name].shape)}",
                )
            )
            continue

        # all good → copy
        model_dict[name] = param
        loaded_keys.append(name)

    # load the new state dict into the model
    model.load_state_dict(model_dict)

    # report
    print(f"✔ Loaded {len(loaded_keys)} parameters")
    for name, reason in skipped:
        print(f"⚠️  Skipped '{name}': {reason}")


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
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
