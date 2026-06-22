"""
Evaluate a trained CoNeMOS (or plain UNet) model on any split.

Computes per-subject, per-structure Dice over annotated channels only,
aggregates by protocol, and saves a CSV + prints a summary table.
Optionally saves predicted segmentations as NIfTI files.

Usage:
    python segsynthtemplate/evaluate_conemos.py \
        experiment=mthesis/train_conemos_real \
        ckpt_path=/path/to/epoch_XXX.ckpt

Optional overrides:
    eval_split=test_i          # which split column value to use (default: test_i)
    save_preds=true            # save predicted NIfTI files (default: false)
    output_dir=eval_results    # where to write CSV + predictions (default: eval_results)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hydra
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from segsynthtemplate.models.conemos_segmentor import CoNeMOSSegmentor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dice_per_channel(
    logits: torch.Tensor,   # (1, num_fg+1, H, W, D)
    labels: torch.Tensor,   # (1, 1,        H, W, D)  integer channel indices
    num_fg: int,
    eps: float = 1e-5,
) -> dict[int, float]:
    """Hard Dice per foreground channel (NaN when structure absent in GT)."""
    pred = logits.argmax(dim=1, keepdim=True)
    out: dict[int, float] = {}
    for c in range(1, num_fg + 1):
        pred_c = (pred == c).float().view(-1)
        tgt_c  = (labels == c).float().view(-1)
        denom  = pred_c.sum() + tgt_c.sum()
        if denom < 1:
            out[c] = float("nan")
        else:
            out[c] = (2.0 * (pred_c * tgt_c).sum() / (denom + eps)).item()
    return out


def _hd95_per_channel(
    logits: torch.Tensor,   # (1, num_fg+1, H, W, D)
    labels: torch.Tensor,   # (1, 1,        H, W, D)  integer channel indices
    num_fg: int,
    spacing: tuple = (0.5, 0.5, 0.5),
) -> dict[int, float]:
    """HD95 (mm) per foreground channel (NaN when structure absent in pred or GT)."""
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, D)
    tgt  = labels.squeeze().cpu().numpy()                 # (H, W, D)
    out: dict[int, float] = {}
    for c in range(1, num_fg + 1):
        pred_c = pred == c
        tgt_c  = tgt  == c
        if pred_c.sum() < 1 or tgt_c.sum() < 1:
            out[c] = float("nan")
            continue
        pred_surf = pred_c ^ binary_erosion(pred_c)
        tgt_surf  = tgt_c  ^ binary_erosion(tgt_c)
        d_pred = distance_transform_edt(~tgt_c,  sampling=spacing)[pred_surf]
        d_tgt  = distance_transform_edt(~pred_c, sampling=spacing)[tgt_surf]
        out[c] = float(np.percentile(np.concatenate([d_pred, d_tgt]), 95))
    return out


def _get_affine(meta_tensor) -> np.ndarray:
    """Extract a (4, 4) affine from a MONAI MetaTensor, handling device and shape quirks."""
    try:
        raw = meta_tensor.meta["affine"]
        if isinstance(raw, torch.Tensor):
            arr = raw.cpu().numpy()
        else:
            arr = np.asarray(raw)
        arr = np.squeeze(arr)
        if arr.shape == (4, 4):
            return arr
    except Exception:
        pass
    return np.diag([0.5, 0.5, 0.5, 1.0])


def _resample_to_orig(
    vol: np.ndarray,
    src_affine: np.ndarray,
    orig_path: str,
    order: int = 0,          # 0 = nearest (segmentation), 1 = linear (image)
) -> "tuple[np.ndarray, np.ndarray]":
    """Resample vol from src_affine space back to the space of the original NIfTI file.

    Returns (resampled_array, orig_affine).
    Uses scipy.ndimage.affine_transform with the exact voxel-to-voxel mapping derived
    from the two affines, so no external resampling library is needed.
    """
    import nibabel as nib
    orig_nii    = nib.load(orig_path)
    orig_affine = orig_nii.affine
    orig_shape  = orig_nii.shape[:3]

    # Mapping: orig voxel → world → src voxel
    # M = inv(src_affine) @ orig_affine   gives src_voxel = M @ orig_voxel
    M = np.linalg.inv(src_affine) @ orig_affine

    resampled = scipy.ndimage.affine_transform(
        vol.astype(np.float64),
        M[:3, :3],
        offset=M[:3, 3],
        output_shape=orig_shape,
        order=order,
        cval=0,
        prefilter=False,
    )
    return resampled, orig_affine


def _print_summary(df: pd.DataFrame, ch_cols: list[str], ch_names: dict[int, str]) -> None:
    hd_cols = [c.replace("dice_ch", "hd95_ch") for c in ch_cols]
    sep = "=" * 84
    print(f"\n{sep}")
    print("SUMMARY BY PROTOCOL")
    print(sep)
    for proto in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == proto]
        print(f"\n  Protocol: {proto}  (n={len(sub)} subjects)")
        print(f"  {'Structure':<26} {'Dice mean':>10}  {'Dice std':>9}  {'HD95 mean':>10}  {'HD95 std':>9}")
        print(f"  {'-' * 68}")
        for dice_col in ch_cols:
            ch_idx   = int(dice_col.split("_")[-1])
            name     = ch_names.get(ch_idx, dice_col)
            hd_col   = f"hd95_ch_{ch_idx:02d}"
            d_vals   = sub[dice_col].dropna()
            h_vals   = sub[hd_col].dropna() if hd_col in sub.columns else pd.Series(dtype=float)
            if len(d_vals) == 0:
                continue
            h_mean = f"{h_vals.mean():>9.3f}" if len(h_vals) else "       n/a"
            h_std  = f"{h_vals.std():>8.3f}"  if len(h_vals) else "      n/a"
            print(f"  {name:<26} {d_vals.mean():>10.4f}  {d_vals.std():>9.4f}  {h_mean}  {h_std}")
        print(f"  {'── mean (annotated) ──':<26} {sub['mean_dice'].mean():>10.4f}  {sub['mean_dice'].std():>9.4f}  {sub['mean_hd95'].mean():>9.3f}  {sub['mean_hd95'].std():>8.3f}")
    print(f"\n{sep}")
    print(f"  OVERALL  (n={len(df)})  Dice = {df['mean_dice'].mean():.4f} ± {df['mean_dice'].std():.4f}   HD95 = {df['mean_hd95'].mean():.3f} ± {df['mean_hd95'].std():.3f} mm")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    ckpt_path             = cfg.get("ckpt_path")
    output_dir            = Path(cfg.get("output_dir", "eval_results"))
    eval_split            = cfg.get("eval_split", "test_i")
    save_preds            = cfg.get("save_preds", False)
    conditioning_protocol = cfg.get("conditioning_protocol", None)  # override FiLM conditioning
    swap_conditioning     = cfg.get("swap_conditioning", False)      # per-sample: use the other protocol
    pathological_only     = cfg.get("pathological_only", None)       # "p", "n", or None (all)

    if conditioning_protocol and swap_conditioning:
        raise ValueError("conditioning_protocol and swap_conditioning are mutually exclusive.")

    if not ckpt_path:
        raise ValueError("Provide ckpt_path=<path/to/checkpoint.ckpt> on the command line.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable structure names — edit to match your label_map_multiprotocol.csv
    ch_names: dict[int, str] = {
        1: "CSF",
        2: "Cortical Grey Matter",
        3: "White Matter",
        4: "ventricles",
        5: "cerebellum",
        6: "deep grey matter",
        7: "brainstem and spinal cord",
        8: "hippocampi and amygdala",
    }

    # ── model ────────────────────────────────────────────────────────────────
    # Prefer load_from_checkpoint so that training-time hyperparameters
    # (including protocol_names, which determines annotation_mask shape) are
    # restored from the checkpoint itself rather than from the current config.
    # This matters for specialist models whose protocol list differs from the
    # joint config that shares the same model YAML.
    # Prefer load_from_checkpoint so that training-time hyperparameters
    # (including protocol_names, which determines annotation_mask shape) are
    # restored from the checkpoint itself rather than from the current config.
    # This matters for specialist models whose protocol list differs from the
    # joint config that shares the same model YAML.
    print(f"\nLoading checkpoint:\n  {ckpt_path}")
    try:
        # Override label_map_csv so the checkpoint's saved cluster path is
        # replaced with the local one from the current config.
        # All other hparams (including protocol_names) come from the checkpoint.
        model = CoNeMOSSegmentor.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            label_map_csv=cfg.model.label_map_csv,
        )
        print("  Loaded via Lightning (training-time hyperparameters)")
    except Exception as e:
        print(f"  Lightning load failed ({e}); falling back to config instantiation")
        model = hydra.utils.instantiate(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_fg      = model.num_fg_channels
    proto_names = model.protocol_names
    ann_mask_all = model.annotation_mask.to(device)   # (P, num_fg) bool

    # ── splits ───────────────────────────────────────────────────────────────
    # eval_split may be a comma-separated list, e.g. "val,test_i"
    eval_splits = [s.strip() for s in str(eval_split).split(",")]

    # ── save_preds setup ─────────────────────────────────────────────────────
    if save_preds:
        import nibabel as nib
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    # ── pathological map ─────────────────────────────────────────────────────
    # Built before inference so we can skip subjects early in the loop.
    pathological_map: dict[str, str] = {}
    for ds_cfg in cfg.data.datasets:
        try:
            split_df = pd.read_csv(ds_cfg["split_file"])
            if "pathological" in split_df.columns:
                pathological_map.update(
                    dict(zip(split_df["participant_id"].astype(str),
                             split_df["pathological"].astype(str)))
                )
        except Exception:
            pass
    if pathological_only is not None:
        keep = {s for s, v in pathological_map.items() if v == pathological_only}
        print(f"  Subject filter: pathological='{pathological_only}' "
              f"→ {len(keep)} matching subjects in split CSVs")

    # ── inference ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    proto_names_list = list(proto_names)
    unknown_protos: set[str] = set()

    # Build a fixed conditioning override if requested (mismatched-protocol eval).
    # The model receives this vector for every sample regardless of the data protocol.
    forced_proto_vec: torch.Tensor | None = None
    if conditioning_protocol is not None:
        if conditioning_protocol not in proto_names_list:
            raise ValueError(
                f"conditioning_protocol='{conditioning_protocol}' not in model's "
                f"training protocols: {proto_names_list}"
            )
        c_idx = proto_names_list.index(conditioning_protocol)
        forced_proto_vec = torch.zeros(len(proto_names_list), device=device)
        forced_proto_vec[c_idx] = 1.0
        print(f"  Conditioning override: '{conditioning_protocol}' "
              f"(index {c_idx} in {proto_names_list})")

    with torch.no_grad():
        for split in eval_splits:
            cfg.data.test_split = split
            datamodule  = hydra.utils.instantiate(cfg.data)
            test_loader = datamodule.test_dataloader()
            n_total     = len(datamodule.test_ds)

            # Count subjects that will actually be evaluated after the pathological filter.
            if pathological_only is not None:
                n_eval = sum(
                    1 for ds_cfg in cfg.data.datasets
                    for _, srow in pd.read_csv(ds_cfg["split_file"]).iterrows()
                    if str(srow.get("splits", "")) == split
                    and pathological_map.get(str(srow["participant_id"]), "n") == pathological_only
                )
                print(f"Split '{split}': {n_total} subjects total, "
                      f"{n_eval} with pathological='{pathological_only}'")
            else:
                n_eval = n_total
                print(f"Split '{split}': {n_eval} subjects")

            with tqdm(total=n_eval, desc=f"Inference [{split}]") as pbar:
                for batch in test_loader:
                    # Extract names before moving tensors to GPU so we can skip cheaply.
                    pnames = batch["protocol_name"]
                    names  = batch["name"]
                    if isinstance(pnames, str):
                        pnames = [pnames]
                    if isinstance(names, str):
                        names = [names]

                    # Skip batch before GPU transfer if subject doesn't match filter.
                    if pathological_only is not None:
                        if not any(pathological_map.get(n, "n") == pathological_only
                                   for n in names):
                            continue

                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    B      = images.shape[0]

                    if forced_proto_vec is not None:
                        protocol_vec = forced_proto_vec.unsqueeze(0).expand(B, -1)
                        cond_names   = [conditioning_protocol] * B
                    elif swap_conditioning:
                        vecs = []
                        cond_names = []
                        for pname_b in pnames:
                            if pname_b in proto_names_list:
                                cur = proto_names_list.index(pname_b)
                                other_idx = next(i for i in range(len(proto_names_list)) if i != cur)
                            else:
                                other_idx = 0
                            v = torch.zeros(len(proto_names_list), device=device)
                            v[other_idx] = 1.0
                            vecs.append(v)
                            cond_names.append(proto_names_list[other_idx])
                        protocol_vec = torch.stack(vecs, dim=0)
                    else:
                        protocol_vec = batch["protocol_vec"].to(device)
                        cond_names   = list(pnames)

                    logits = model.net(images, protocol_vec)   # (B, num_fg+1, H, W, D)

                    for b in range(B):
                        pname   = pnames[b]
                        subject = names[b]

                        # Per-sample filter: needed when batch_size > 1 mixes
                        # matching and non-matching subjects in the same batch.
                        if pathological_only is not None:
                            if pathological_map.get(subject, "n") != pathological_only:
                                continue
                        pbar.update(1)

                        if pname in proto_names_list:
                            ann = ann_mask_all[proto_names_list.index(pname)]  # (num_fg,) bool
                        else:
                            # Cross-protocol eval: model not trained on this protocol.
                            # Use all-True mask so all channels are scored.
                            ann = torch.ones(num_fg, dtype=torch.bool, device=device)
                            if pname not in unknown_protos:
                                print(f"\n  [warn] protocol '{pname}' not in model's training protocols "
                                      f"{proto_names_list}; using all-channel mask for this protocol.")
                                unknown_protos.add(pname)

                        ch_dice = _dice_per_channel(logits[b:b+1], labels[b:b+1], num_fg)
                        ch_hd95 = _hd95_per_channel(logits[b:b+1], labels[b:b+1], num_fg)

                        row: dict = {"subject": subject, "protocol": pname, "conditioning": cond_names[b]}
                        for c in range(1, num_fg + 1):
                            row[f"dice_ch_{c:02d}"] = ch_dice[c] if ann[c - 1] else float("nan")
                            row[f"hd95_ch_{c:02d}"] = ch_hd95[c] if ann[c - 1] else float("nan")

                        annotated_dice = [ch_dice[c] for c in range(1, num_fg + 1)
                                          if ann[c - 1] and not np.isnan(ch_dice[c])]
                        annotated_hd95 = [ch_hd95[c] for c in range(1, num_fg + 1)
                                          if ann[c - 1] and not np.isnan(ch_hd95[c])]
                        row["mean_dice"] = float(np.mean(annotated_dice)) if annotated_dice else float("nan")
                        row["mean_hd95"] = float(np.mean(annotated_hd95)) if annotated_hd95 else float("nan")
                        rows.append(row)

                        if save_preds:
                            # ── transformed space (same grid as dice computation) ────
                            affine  = _get_affine(labels[b])
                            pred_np = logits[b:b+1].argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)
                            gt_np   = labels[b].squeeze().cpu().numpy().astype(np.int16)
                            img_np  = images[b].squeeze().cpu().numpy().astype(np.float32)
                            nib.save(nib.Nifti1Image(pred_np, affine),
                                     pred_dir / f"{subject}_{pname}_pred.nii.gz")
                            nib.save(nib.Nifti1Image(gt_np,   affine),
                                     pred_dir / f"{subject}_{pname}_gt.nii.gz")
                            nib.save(nib.Nifti1Image(img_np,  affine),
                                     pred_dir / f"{subject}_{pname}_img.nii.gz")

                            # ── original space (aligned with raw BIDS files) ─────────
                            orig_path = None
                            try:
                                raw_fn = images[b].meta.get("filename_or_obj", None)
                                if isinstance(raw_fn, (list, tuple)):
                                    raw_fn = raw_fn[0]
                                if raw_fn and Path(str(raw_fn)).exists():
                                    orig_path = str(raw_fn)
                            except Exception:
                                pass

                            if orig_path is not None:
                                try:
                                    pred_orig, orig_aff = _resample_to_orig(pred_np, affine, orig_path, order=0)
                                    gt_orig,   _        = _resample_to_orig(gt_np,   affine, orig_path, order=0)
                                    nib.save(nib.Nifti1Image(pred_orig.astype(np.int16), orig_aff),
                                             pred_dir / f"{subject}_{pname}_pred_orig.nii.gz")
                                    nib.save(nib.Nifti1Image(gt_orig.astype(np.int16),   orig_aff),
                                             pred_dir / f"{subject}_{pname}_gt_orig.nii.gz")
                                except Exception as e:
                                    print(f"\n  [warn] original-space resample failed for {subject}: {e}")

    # ── pathology join ───────────────────────────────────────────────────────
    if pathological_map:
        for row in rows:
            row["pathological"] = pathological_map.get(row["subject"], None)

    # ── save + print ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    ch_cols = [c for c in df.columns if c.startswith("dice_ch_")]

    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-subject results → {csv_path}")

    _print_summary(df, ch_cols, ch_names)


if __name__ == "__main__":
    main()
