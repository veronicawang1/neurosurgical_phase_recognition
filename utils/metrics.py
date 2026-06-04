import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def edit_distance(pred: list, gt: list) -> float:
    """
    Normalized edit distance on segment sequences (not frame-level).
    Returns score in [0, 1] where 1 is perfect.
    """
    pred_seg = _to_segments(pred)
    gt_seg = _to_segments(gt)

    n, m = len(pred_seg), len(gt_seg)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_seg[i - 1] == gt_seg[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    max_len = max(n, m) if max(n, m) > 0 else 1
    return 1.0 - dp[n][m] / max_len


def segmental_f1(pred: list, gt: list, overlap_thresholds=(0.1, 0.25, 0.5)):
    """
    Segmental F1 at multiple overlap thresholds.
    Returns dict: {thresh: f1_score}
    """
    pred_segs = _to_segment_intervals(pred)
    gt_segs = _to_segment_intervals(gt)
    results = {}

    for thresh in overlap_thresholds:
        tp, fp = 0, 0
        gt_matched = [False] * len(gt_segs)

        for p_label, p_start, p_end in pred_segs:
            matched = False
            for j, (g_label, g_start, g_end) in enumerate(gt_segs):
                if gt_matched[j] or p_label != g_label:
                    continue
                intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
                union = max(p_end, g_end) - min(p_start, g_start)
                iou = intersection / union if union > 0 else 0
                if iou >= thresh:
                    tp += 1
                    gt_matched[j] = True
                    matched = True
                    break
            if not matched:
                fp += 1

        fn = sum(1 for m in gt_matched if not m)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[thresh] = round(f1, 4)

    return results


def confusion_matrix(pred: list, gt: list, num_classes: int):
    """Returns num_classes x num_classes confusion matrix (rows=gt, cols=pred)."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, g in zip(pred, gt):
        if g >= 0:
            matrix[g][p] += 1
    return matrix


def _to_segments(labels: list) -> list:
    if not labels:
        return []
    segs = [labels[0]]
    for l in labels[1:]:
        if l != segs[-1]:
            segs.append(l)
    return segs


def _to_segment_intervals(labels: list) -> list:
    if not labels:
        return []
    segs = []
    cur_label, cur_start = labels[0], 0
    for i, l in enumerate(labels[1:], 1):
        if l != cur_label:
            segs.append((cur_label, cur_start, i))
            cur_label, cur_start = l, i
    segs.append((cur_label, cur_start, len(labels)))
    return segs
