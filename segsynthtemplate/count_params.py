import sys
sys.path.insert(0, "/home/danielaco/Documents/segsynthtemplate")

import rootutils
rootutils.setup_root("/home/danielaco/Documents/segsynthtemplate", indicator=".project-root", pythonpath=True)

from segsynthtemplate.models.conemos_unet import CoNeMOSUNet, FiLM, CondMLP
import torch.nn as nn

#python segsynthtemplate/count_params.py

def count(module):
    return sum(p.numel() for p in module.parameters())

def breakdown(net: CoNeMOSUNet, label: str):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    # MLP
    mlp = count(net.cond_mlp) if net.conditioned else 0
    print(f"  CondMLP:          {mlp:>10,}")

    # FiLM layers (inside encoder, decoder, final)
    film_params = 0
    for block in list(net.encoder) + list(net.decoder):
        for film in block.films:
            if isinstance(film, FiLM):
                film_params += count(film)
    if isinstance(net.final_film, FiLM):
        film_params += count(net.final_film)
    print(f"  FiLM projections: {film_params:>10,}")

    # Conv weights (no bias)
    conv_weights = sum(
        p.numel() for m in net.modules()
        if isinstance(m, nn.Conv3d)
        for name, p in m.named_parameters()
        if "weight" in name
    )
    print(f"  Conv weights:     {conv_weights:>10,}")

    # Conv biases
    conv_biases = sum(
        p.numel() for m in net.modules()
        if isinstance(m, nn.Conv3d)
        for name, p in m.named_parameters()
        if "bias" in name
    )
    print(f"  Conv biases:      {conv_biases:>10,}")

    # Norms (if any)
    norm_params = sum(
        p.numel() for m in net.modules()
        if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))
        for p in m.parameters()
    )
    if norm_params:
        print(f"  Norm params:      {norm_params:>10,}")

    total = count(net)
    print(f"  {'─'*30}")
    print(f"  TOTAL:            {total:>10,}")

# --- Conditioned ---
conditioned = CoNeMOSUNet(
    num_protocols=2, num_fg_channels=8,
    n_levels=4, n_features_init=16, feat_mult=2,
    n_conv_per_level=2, conv_size=3, norm_type=None,
    n_conditioned_layers=0, mlp_hidden_dim=64, mlp_n_layers=4,
    conditioned=True,
)

# --- Unconditioned ---
unconditioned = CoNeMOSUNet(
    num_protocols=2, num_fg_channels=8,
    n_levels=4, n_features_init=16, feat_mult=2,
    n_conv_per_level=2, conv_size=3, norm_type=None,
    n_conditioned_layers=0, mlp_hidden_dim=64, mlp_n_layers=4,
    conditioned=False,
)

breakdown(conditioned,   "CoNeMOS (conditioned=True)")
breakdown(unconditioned, "Plain UNet (conditioned=False)")
print(f"\n  Difference: {count(conditioned) - count(unconditioned):+,}")
