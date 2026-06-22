"""CoNeMOS LightningModule — protocol-conditioned variant.

One forward pass per image, conditioned on the protocol one-hot vector
(batch["protocol_vec"]).  The network predicts all foreground channels at once;
the Dice loss is masked to the channels annotated by that protocol.

The annotation mask is derived at init time from label_map_csv so that
channels a protocol does not label are excluded from the loss, even when no
positive voxels happen to be present in a particular volume.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ann_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss with protocol-conditioned softmax masking.

    Unannotated fg channel logits are set to -inf before CE computes its
    internal softmax, so the probability mass is only distributed among the
    classes this protocol actually annotates.  Background (channel 0) is
    always included.

    Args:
        logits:   (B, num_fg_channels + 1, H, W, D) raw network output.
        labels:   (B, 1, H, W, D) integer targets; 0 = background.
        ann_mask: (B, num_fg_channels) bool — True = channel is annotated.
    Returns:
        Scalar CE loss.
    """
    masked = logits.clone()
    # logits[:, 0] = background (always valid); logits[:, 1:] = fg channels
    # ann_mask[:, c] corresponds to logits[:, c + 1]
    unannotated = ~ann_mask                            # (B, num_fg_channels)
    masked[:, 1:][unannotated] = float("-inf")
    return F.cross_entropy(masked, labels.squeeze(1).long())


def _dice_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ann_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Hard Dice over annotated fg channels, computed from argmax of logits.

    Args:
        logits:   (B, num_fg_channels + 1, H, W, D).
        labels:   (B, 1, H, W, D) integer targets.
        ann_mask: (B, num_fg_channels) bool.
    Returns:
        Scalar mean Dice over annotated (sample, channel) pairs.
    """
    preds = logits.argmax(dim=1, keepdim=True)   # (B, 1, H, W, D)
    B, num_fg = ann_mask.shape
    dice_sum   = torch.tensor(0.0, device=logits.device)
    count      = torch.tensor(0,   device=logits.device)
    for c in range(1, num_fg + 1):                # fg channels are 1-indexed
        mask_c = ann_mask[:, c - 1]               # (B,) bool
        if not mask_c.any():
            continue
        pred_c = (preds[mask_c] == c).float().view(mask_c.sum(), -1)
        tgt_c  = (labels[mask_c] == c).float().view(mask_c.sum(), -1)
        inter  = (pred_c * tgt_c).sum(-1)
        denom  = pred_c.sum(-1) + tgt_c.sum(-1)
        dice_sum += (2.0 * inter / (denom + eps)).sum()
        count    += mask_c.sum()
    return dice_sum / count.clamp(min=1)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class CoNeMOSSegmentor(LightningModule):
    """Protocol-conditioned CoNeMOS training / validation module.

    Args:
        net: ``CoNeMOSUNet`` instance.
        optimizer: Partial Adam constructor.
        scheduler: Partial LR-scheduler constructor, or ``None``.
        compile: Whether to ``torch.compile`` the network at fit start.
        lr: Learning rate.
        num_fg_channels: Number of foreground output channels (max channel in
            label_map_csv, excluding background=0).
        protocol_names: Ordered list of protocol names matching
            ``MultiProtocolDataset.protocol_registry`` (first-seen order in
            the datamodule's ``datasets`` list).
        label_map_csv: Path to the CSV with columns ``protocol``,
            ``raw_label``, ``channel``.  Used to build the per-protocol
            annotation mask at init time.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        compile: bool,
        lr: float,
        num_fg_channels: int,
        protocol_names: List[str],
        label_map_csv: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net             = net
        self.num_fg_channels = num_fg_channels
        self.protocol_names  = protocol_names

        # annotation_mask[p, c] = True when protocol p annotates fg channel c+1
        self.register_buffer(
            "annotation_mask",
            self._build_annotation_mask(label_map_csv, protocol_names, num_fg_channels),
        )

        self.train_loss    = MeanMetric()
        self.train_dice    = MeanMetric()
        self.val_loss      = MeanMetric()
        self.val_dice      = MeanMetric()
        self.val_dice_best = MaxMetric()
        # per-protocol dice tracked separately so WandB shows per-dataset curves
        self.val_dice_per_protocol = torch.nn.ModuleDict({
            name: MeanMetric() for name in protocol_names
        })

    # ------------------------------------------------------------------
    # Init helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_annotation_mask(
        label_map_csv: str,
        protocol_names: List[str],
        num_fg_channels: int,
    ) -> torch.Tensor:
        """Return bool tensor (num_protocols, num_fg_channels).

        True where protocol p annotates foreground channel c (1-indexed).
        """
        df   = pd.read_csv(label_map_csv)
        mask = torch.zeros(len(protocol_names), num_fg_channels, dtype=torch.bool)
        for p_idx, name in enumerate(protocol_names):
            rows = df[df["protocol"] == name]
            for _, row in rows.iterrows():
                ch = int(row["channel"])
                if ch > 0:                      # skip background (ch == 0)
                    mask[p_idx, ch - 1] = True
        return mask

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(x, condition)

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Return per-channel softmax probability maps.

        Channel 0 = background probability; channels 1..num_fg_channels =
        foreground structure probabilities.  Callers can do
        ``probs.argmax(dim=1)`` for a hard label map.
        """
        images       = batch["image"]         # (B, 1, H, W, D)
        protocol_vec = batch["protocol_vec"]  # (B, num_protocols)
        logits = self.net(images, protocol_vec)          # (B, num_fg_channels+1, H, W, D)
        probs  = torch.softmax(logits, dim=1)            # (B, num_fg_channels+1, H, W, D)
        return {"probs": probs, "names": batch.get("name", [])}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_dice.reset()
        self.val_dice_best.reset()
        for m in self.val_dice_per_protocol.values():
            m.reset()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        images       = batch["image"]         # (B, 1, H, W, D)
        labels       = batch["label"]         # (B, 1, H, W, D)  integer channels
        protocol_vec = batch["protocol_vec"]  # (B, num_protocols) one-hot

        proto_idx = protocol_vec.argmax(dim=1)           # (B,)
        ann_mask  = self.annotation_mask[proto_idx]      # (B, num_fg_channels) bool

        logits = self.net(images, protocol_vec)          # (B, num_fg_channels+1, H, W, D)
        loss   = _masked_ce_loss(logits, labels, ann_mask)

        self.train_loss(loss)
        self.train_dice(_dice_from_logits(logits.detach(), labels, ann_mask))
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        images       = batch["image"]
        labels       = batch["label"]
        protocol_vec = batch["protocol_vec"]

        proto_idx = protocol_vec.argmax(dim=1)
        ann_mask  = self.annotation_mask[proto_idx]

        logits = self.net(images, protocol_vec)

        self.val_loss(_masked_ce_loss(logits, labels, ann_mask))
        self.val_dice(_dice_from_logits(logits, labels, ann_mask))

        pnames = batch["protocol_name"]
        if isinstance(pnames, str):
            pnames = [pnames]
        for b, pname in enumerate(pnames):
            dice_b = _dice_from_logits(logits[b:b+1], labels[b:b+1], ann_mask[b:b+1])
            self.val_dice_per_protocol[pname](dice_b)

    def on_validation_epoch_end(self) -> None:
        dice = self.val_dice.compute()
        self.val_dice_best(dice)
        self.log("val/loss",      self.val_loss.compute(),     prog_bar=True, sync_dist=True)
        self.log("val/dice",      dice,                         prog_bar=True, sync_dist=True)
        self.log("val/dice_best", self.val_dice_best.compute(), prog_bar=True, sync_dist=True)
        for pname, metric in self.val_dice_per_protocol.items():
            self.log(f"val/dice_{pname}", metric.compute(), prog_bar=False, sync_dist=True)
            metric.reset()
        self.val_loss.reset()
        self.val_dice.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        images       = batch["image"]
        labels       = batch["label"]
        protocol_vec = batch["protocol_vec"]

        proto_idx = protocol_vec.argmax(dim=1)
        ann_mask  = self.annotation_mask[proto_idx]
        logits    = self.net(images, protocol_vec)

        self.log("test/loss", _masked_ce_loss(logits, labels, ann_mask),
                 on_step=False, on_epoch=True)
        self.log("test/dice", _dice_from_logits(logits, labels, ann_mask),
                 on_step=False, on_epoch=True)

        pnames = batch["protocol_name"]
        if isinstance(pnames, str):
            pnames = [pnames]
        for b, pname in enumerate(pnames):
            dice_b = _dice_from_logits(logits[b:b+1], labels[b:b+1], ann_mask[b:b+1])
            self.log(f"test/dice_{pname}", dice_b, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters(), lr=self.hparams.lr
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   "val/loss",
                    "interval":  "epoch",
                    "frequency": 1,
                    "strict":    False,
                },
            }
        return {"optimizer": optimizer}
