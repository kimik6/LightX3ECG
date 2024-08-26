
import os, sys
import warnings; warnings.filterwarnings("ignore")
import pytorch_lightning; pytorch_lightning.seed_everything(22)

import numpy as np
from sklearn.metrics import f1_score
def thresholds_search(labels, preds):
    search_range = [round(t, 2) for t in np.arange(1, 20)*0.05]

    optimal_thresholds = []
    for cls in range(preds.shape[1]):
        f1_scores_cls = []
        for threshold in search_range:
            labels_cls, preds_cls = labels[:, cls], preds[:, cls]

            preds_cls = list(np.where(preds[:, cls] >= threshold, 1, 0))
            f1_scores_cls.append(f1_score(
                labels_cls, preds_cls
                , average = "macro"
            ))

        optimal_thresholds.append(search_range[np.argmax(f1_scores_cls)])

    return optimal_thresholds