#!/usr/bin/env python3
import argparse
import sys
import torch

def load_checkpoint(path):
    """Load the full checkpoint (might contain 'state_dict' plus other metadata)."""
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def get_state_dict(ckpt):
    """Pull out the actual state_dict from a checkpoint."""
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

def replace_state_dict(ckpt, new_sd):
    """Inject the interpolated state_dict back into the original checkpoint."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = new_sd
        return ckpt
    # if the original was just a bare state_dict, return the bare interpolated one
    return new_sd

def main():
    parser = argparse.ArgumentParser(
        description="Interpolate two model checkpoints weight by weight."
    )
    parser.add_argument("checkpoint1", help="Path to the first checkpoint")
    parser.add_argument("checkpoint2", help="Path to the second checkpoint")
    parser.add_argument("output",      help="Where to save the interpolated checkpoint")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for the first checkpoint (default 0.5)"
    )
    args = parser.parse_args()

    ckpt1 = load_checkpoint(args.checkpoint1)
    ckpt2 = load_checkpoint(args.checkpoint2)
    sd1   = get_state_dict(ckpt1)
    sd2   = get_state_dict(ckpt2)

    if sd1.keys() != sd2.keys():
        print("Error: checkpoints have different state_dict keys.", file=sys.stderr)
        sys.exit(1)

    interpolated = {}
    for k in sd1:
        w1, w2 = sd1[k], sd2[k]
        if w1.size() != w2.size():
            print(
                f"Error: shape mismatch for '{k}': {w1.size()} vs {w2.size()}",
                file=sys.stderr
            )
            sys.exit(1)
        interpolated[k] = args.alpha * w1 + (1.0 - args.alpha) * w2

    # put interpolated state_dict back into checkpoint1’s structure
    out_ckpt = replace_state_dict(ckpt1, interpolated)
    torch.save(out_ckpt, args.output)
    print(f"Interpolated checkpoint saved to {args.output}")

if __name__ == "__main__":
    main()
