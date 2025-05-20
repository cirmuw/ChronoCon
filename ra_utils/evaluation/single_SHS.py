import pandas as pd
import numpy as np
from typing import List

from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)


def combine_predictions(
    sources: List[str],
    keys_for_df: List[str] = [
        'labels',
        'preds',
        # 'probs',
        'file_name',
        'score_type',
        'JSN_or_ERO',
        'extremity',
        'patient_id',
    ],
    check_for_score_type_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Combine predictions from different sources into a single DataFrame.
    """
    data = []
    seen_score_types = set()
    for src in sources:
        data_ = np.load(src)
        if keys_for_df is not None:
            data_ = pd.DataFrame({k: data_[k] for k in keys_for_df})
        data.append(data_)
        if check_for_score_type_duplicates:
            score_types = data_["score_type"].unique()
            for st in score_types:
                if st in seen_score_types:
                    raise ValueError(f"Duplicate score type found: {st}")
                seen_score_types.add(st)

    df = pd.concat(data, ignore_index=True)
    return df


def get_main_metrics(df, allowed_classes = ['0', '1', '2', '3', '4', '5'], index_name = "erosion foot"):
    preds = df["preds"].astype(int).to_numpy()
    labels = df["labels"].astype(int).to_numpy()

    # exclude surgery class from evaluation
    m = labels < len(allowed_classes)
    preds = preds[m]
    labels = labels[m]


    metrics = calculate_some_classification_metrics(
        preds,  
        labels,     
        calc_ICC3=0, 
        add_support=1,
    )

    report_dct = classification_report(
        df["labels"].astype(int).to_numpy(),
        df["preds"].astype(int).to_numpy(),
        output_dict=True,
        #zero_division=0
    )
    report = pd.DataFrame(report_dct).transpose()
    report = report[["precision", "recall", "f1-score", "support"]]


    # Add macro average over subset of classes
    
    max_class = allowed_classes[-1]
    report_allowed = report.loc[allowed_classes]
    name = f"macro avg 0-{max_class}"
    report.loc[name] = report_allowed.mean()
    report.loc[name, "support"] = report_allowed["support"].sum()
    report["support"] = report["support"].astype(int)
    part_macro_avg_f1 = report.loc[name, "f1-score"]


    metrics_df = pd.DataFrame([metrics], index=[index_name])
    metrics_df = metrics_df[['n_samples eval', 'accuracy', 'error > 1 (percent)', 'balanced acc.', 'balanced acc. (error < 2)', 'rmse']]
    metrics_df = metrics_df.rename(columns={"n_samples eval": "joints"})
    metrics_df[f"{name} f1"] = part_macro_avg_f1

    return metrics_df, report
