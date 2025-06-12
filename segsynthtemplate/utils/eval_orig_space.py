from pathlib import Path
import pandas as pd
import monai
import torch
import SimpleITK as sitk
from tqdm import tqdm
from monai.metrics import DiceMetric, compute_hausdorff_distance
import argparse
import numpy as np

# one hot encode the labels
from monai.transforms import AsDiscrete


# ARGPARSE
parser = argparse.ArgumentParser(description="Evaluate metrics in original space")
parser.add_argument(
    "--bids_path", type=str, required=True, help="Path to BIDS dataset root"
)
parser.add_argument(
    "--preds_path", type=str, required=True, help="Path to predictions directory"
)
parser.add_argument(
    "--gt_suffix",
    type=str,
    default="dseg",
    help="Suffix for ground-truth segmentation files",
)
parser.add_argument(
    "--pred_suffix",
    type=str,
    default="pred",
    help="Suffix for prediction files (default: 'pred')",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=None,
    help="Number of classes (if not provided, inferred from data)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Computation device (e.g., 'cuda' or 'cpu')",
)
parser.add_argument(
    "--hd95",
    action="store_true",
    default=True,
    help="Compute the 95th percentile Hausdorff distance",
)


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    bids_path = args.bids_path
    preds_path = args.preds_path
    gt_suffix = args.gt_suffix
    pred_suffix = args.pred_suffix
    num_classes = args.num_classes
    device = args.device
    hd95 = args.hd95

    # main
    pred_subjects = list(
        Path(preds_path).glob("sub*/**/sub-*_{}*.nii.gz".format(pred_suffix))
    )

    out_path = Path(preds_path) / "metrics_orig_space.csv"
    bids_path = Path(bids_path)
    preds_path = Path(preds_path)

    testsplit_df = []
    for subj in tqdm(pred_subjects):

        predpath = subj
        ses = predpath.name.split("_")[1]  # assuming session is in the filename

        if "ses" not in ses:
            ses = None

        subjname = predpath.name.split("_")[0]  # assuming subject is in the filename

        if ses is not None:
            gt_path = list(
                bids_path.glob(
                    f"{subjname}/{ses}/anat/{subjname}_{ses}*_{gt_suffix}.nii.gz"
                )
            )[0]
        else:
            gt_path = list(
                bids_path.glob(f"{subjname}/anat/{subjname}*_{gt_suffix}.nii.gz")
            )[0]

        # read the images
        pred = sitk.ReadImage(str(predpath))
        # get spacing
        image_resolution = pred.GetSpacing()
        pred = sitk.GetArrayFromImage(pred)
        pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(torch.int64)
        pred = pred.to(device)
        label = sitk.ReadImage(str(gt_path))
        label = sitk.GetArrayFromImage(label)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).to(torch.int64)
        label = label.to(device)
        if num_classes is None:
            num_classes = int(label.max()) + 1
        # print(num_classes)
        test_split = preds_path.name
        exp_name = str(preds_path).split("/")[-3]

        onehoteencoder = AsDiscrete(to_onehot=num_classes)

        dice_metric = DiceMetric(
            include_background=False,
        )
        pred_1h = onehoteencoder(pred)
        gt_1h = onehoteencoder(label)
        # calculate dice
        dice = (
            dice_metric(
                pred_1h,
                gt_1h,
            )
            .cpu()
            .numpy()
        )[:]

        hd95th = (
            (
                compute_hausdorff_distance(
                    pred_1h,
                    gt_1h,
                    percentile=95,
                    include_background=False,
                    spacing=image_resolution,
                )
                .cpu()
                .numpy()
            )
            if hd95
            else None
        )
        mean_dice = dice[:].mean()
        subj_res = [
            {
                "Metric": "DSC",
                "Value": dice[x][0],
                "subj": subjname,
                "Label": x,
                "Split": test_split,
                "Exp": exp_name,
                "Session": ses,  # assuming session is in the filename
            }
            for x in range(1, len(dice))
        ]
        subj_res_hd = (
            [
                {
                    "Metric": "HD95",
                    "Value": hd95th[x][0],
                    "subj": subjname,
                    "Label": x,
                    "Split": test_split,
                    "Exp": exp_name,
                    "Session": ses,
                }
                for x in range(1, len(hd95th))
            ]
            if hd95
            else []
        )

        # save GT and pred volumes and their volume similarity
        subj_res_vs = []
        for lab in range(1, num_classes):
            lab_volume = np.sum(label.cpu().numpy() == lab)
            pred_volume = np.sum(pred.cpu().numpy() == lab)

            # scale by voxel size
            lab_volume *= np.prod(image_resolution)
            pred_volume *= np.prod(image_resolution)

            subj_res_vs.append(
                {
                    "Metric": "VS_GT_Pred",
                    "Value": [lab_volume, pred_volume],
                    "subj": subjname,
                    "Label": lab,
                    "Split": test_split,
                    "Exp": exp_name,
                    "Session": ses,  # assuming session is in the filename
                }
            )

        testsplit_df.extend(subj_res + subj_res_hd + subj_res_vs)

    testsplit_df = pd.DataFrame(testsplit_df)

    testsplit_df.to_csv(out_path, index=False)
