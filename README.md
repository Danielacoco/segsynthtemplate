# segsynthtemplate

First create the environment with the following command:

```bash
conda env create -f environment.yml
```

Then activate the environment with the following command:

```bash
conda activate segsynthtemplate
```
Then instal the fetalsyngen package 

```bash
pip install -r requirements.txt
```



## Evaluation and Inference

Use ``segsynthtemplate/inference.py`` to run the inference script. The inference parameters can be set in the ``configs/inference.yaml`` file. Use this script for running model inference on a **test dataset when ground truth is not available**. It will save the results in the original image space.

When ground truth is available, use ``segsynthtemplate/eval.py``. The evaluation parameters can be set in the ``configs/eval.yaml`` file. This script will compute the evaluation metrics and save them in a pandas dataframe in the results directory, along with the predicted images. The metrics will be calculated in the **resampled image space** but the resulting segmentations will be saved in the original image space as well.

To calculate the metrics in the original image space, you can use the ``segsynthtemplate/utils/eval_orig_space.py`` script. Run it after the ``segsynthtemplate/eval.py`` to obtain additional csv file with the metrics calculated in the original image space.

```
usage: eval_orig_space.py [-h] --bids_path BIDS_PATH --preds_path PREDS_PATH [--gt_suffix GT_SUFFIX] [--pred_suffix PRED_SUFFIX] [--num_classes NUM_CLASSES]
                          [--device DEVICE] [--hd95]

Evaluate metrics in original space

options:
  -h, --help            show this help message and exit
  --bids_path BIDS_PATH
                        Path to BIDS dataset root
  --preds_path PREDS_PATH
                        Path to predictions directory
  --gt_suffix GT_SUFFIX
                        Suffix for ground-truth segmentation files
  --pred_suffix PRED_SUFFIX
                        Suffix for prediction files (default: 'pred')
  --num_classes NUM_CLASSES
                        Number of classes (if not provided, inferred from data)
  --device DEVICE       Computation device (e.g., 'cuda' or 'cpu')
  --hd95                Compute the 95th percentile Hausdorff distance
```