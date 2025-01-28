import torchmetrics
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
def caluclate_tissue_dsc(path2predictions: str, path2bids: str):

    tissue_classes = {1: 'CSF', 2: 'GM', 3: 'WM', 4: 'LV', 5: 'CBM', 6: 'SGM', 7: 'BS'}
    tissue_dices = []
    dice = torchmetrics.Dice(num_classes=1, multiclass=False)
    predictions = list(Path(path2predictions).glob('sub-*pred*.nii.gz'))
    path2bids = Path(path2bids)
    for pred in tqdm(predictions):
        subj = str(pred).split('/')[-1].split('_')[0]
        gt_segm = str(pred).replace('pred', 'dseg')
        gt_segm = path2bids/f'{subj}/anat/{Path(gt_segm).name}'

        pred = sitk.ReadImage(str(pred))
        gt_segm = sitk.ReadImage(str(gt_segm))
        pred = torch.tensor(sitk.GetArrayFromImage(pred))
        gt_segm = torch.tensor(sitk.GetArrayFromImage(gt_segm))
        for ti, tn in tissue_classes.items():
            pred_tissue = (pred == ti).long()
            gt_tissue = (gt_segm == ti).long()
            pred_tissue = pred_tissue.unsqueeze(0)
            gt_tissue = gt_tissue.unsqueeze(0)
            tissue_dices.append({'subject': subj,
                                 'tissue': tn,
                                 'dice': dice(pred_tissue, gt_tissue).item()})
    tissue_dices = pd.DataFrame(tissue_dices)
    tissue_dices.to_csv(Path(path2predictions)/'tissue_dices.csv')

    return tissue_dices


if __name__ == '__main__':
    # TODO: Move paths to argparse arguemnts
    path2preds = '/media/vzalevskyi/data/fetalSynthSeg/synthSeg/predictions/train_UnetSegm_test_realFetaTrain/dataset_0'
    path2bids = '/media/vzalevskyi/data/FETA_challenge/feta_2.2'

    data = caluclate_tissue_dsc(path2preds,
                                path2bids)

    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="tissue", y="dice", data=data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title('Dice scores for different tissues')
    ax.set_xlabel('Tissue')
    ax.set_ylabel('Dice score')
    sns.despine(offset=10, trim=True)
    plt.savefig(f'{path2preds}/tissue_dices.png')