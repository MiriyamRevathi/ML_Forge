"""
MLForge ML Engine - Detailed Classification Metrics Calculator Module
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
Confusion Matrix, Log Loss, and Per-Class Breakdown Metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
    precision_recall_curve,
    auc
)


class ClassificationMetricsCalculator:
    """
    Comprehensive Classification Metrics Evaluator.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_labels: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes complete suite of classification metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_classes) <= 2
        labels = class_labels if class_labels is not None else unique_classes

        # Base Metrics
        acc = float(accuracy_score(y_true, y_pred))

        avg_mode = "binary" if is_binary else "weighted"
        prec = float(precision_score(y_true, y_pred, average=avg_mode, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=avg_mode, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0))

        # Macro and Micro averages
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # ROC-AUC & PR-AUC
        roc_auc_val = None
        pr_auc_val = None
        log_loss_val = None

        if y_prob is not None:
            try:
                if is_binary:
                    prob_pos = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob
                    roc_auc_val = float(roc_auc_score(y_true, prob_pos))
                    
                    # PR-AUC
                    p_curve, r_curve, _ = precision_recall_curve(y_true, prob_pos)
                    pr_auc_val = float(auc(r_curve, p_curve))

                    # Log Loss
                    log_loss_val = float(log_loss(y_true, y_prob))
                else:
                    roc_auc_val = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                    log_loss_val = float(log_loss(y_true, y_prob))
            except Exception:
                pass

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_list = cm.tolist()

        # Per-class metrics breakdown
        report_dict = classification_report(
            y_true, y_pred,
            target_names=[str(lbl) for lbl in labels],
            output_dict=True,
            zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "accuracy_percentage": round(acc * 100, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
            "pr_auc": round(pr_auc_val, 4) if pr_auc_val is not None else None,
            "log_loss": round(log_loss_val, 4) if log_loss_val is not None else None,
            "is_binary": is_binary,
            "confusion_matrix": {
                "matrix": cm_list,
                "labels": [str(lbl) for lbl in labels]
            },
            "per_class_report": report_dict
        }
