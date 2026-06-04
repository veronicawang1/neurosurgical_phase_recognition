"""
Post-processing for surgical phase predictions.
Both methods improve temporal ordering without retraining.
"""
import numpy as np
import torch


def temporal_smooth(probs: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Median filter over per-frame class probabilities.
    probs: [T, num_classes] softmax probabilities
    window: number of frames to smooth over (odd number recommended)
    Returns smoothed predictions as integer class indices [T].
    """
    T, C = probs.shape
    half = window // 2
    smoothed = np.zeros_like(probs)
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        smoothed[t] = probs[start:end].mean(axis=0)
    return smoothed.argmax(axis=1)


def viterbi_decode(probs: np.ndarray, transition_matrix: np.ndarray = None,
                   num_classes: int = 4) -> np.ndarray:
    """
    Viterbi decoding to enforce valid phase transition sequences.

    probs: [T, num_classes] softmax probabilities (emission probs)
    transition_matrix: [num_classes, num_classes] log transition probs.
        transition_matrix[i, j] = log prob of transitioning from class i to class j.
        If None, uses surgical phase ordering constraint.
    num_classes: number of classes

    Surgical phase order:
        0: Brain Exposure
        1: Parent Vessel ID
        2: Dome and Neck ID
        3: Clipping

    Returns: [T] integer class predictions.
    """
    if transition_matrix is None:
        transition_matrix = _default_transition_matrix(num_classes)

    T = len(probs)
    log_probs = np.log(probs + 1e-10)
    log_trans = transition_matrix

    # dp[t, c] = best log-prob of sequence ending in class c at time t
    dp = np.full((T, num_classes), -np.inf)
    backtrack = np.zeros((T, num_classes), dtype=int)

    dp[0] = log_probs[0]

    for t in range(1, T):
        for c in range(num_classes):
            scores = dp[t - 1] + log_trans[:, c]
            best_prev = np.argmax(scores)
            dp[t, c] = scores[best_prev] + log_probs[t, c]
            backtrack[t, c] = best_prev

    # backtrack
    preds = np.zeros(T, dtype=int)
    preds[T - 1] = np.argmax(dp[T - 1])
    for t in range(T - 2, -1, -1):
        preds[t] = backtrack[t + 1, preds[t + 1]]

    return preds


def _default_transition_matrix(num_classes: int) -> np.ndarray:
    """
    Surgical phase transition matrix.
    Phases progress forward (0→1→2→3) with high probability.
    Staying in same phase is allowed. Going backward is heavily penalized.
    """
    # stay or advance: high prob
    # skip forward: medium penalty
    # go backward: severe penalty
    mat = np.full((num_classes, num_classes), -10.0)  # default: very unlikely

    for i in range(num_classes):
        mat[i, i] = np.log(0.95)           # stay in same phase
        if i + 1 < num_classes:
            mat[i, i + 1] = np.log(0.04)  # advance one phase
        if i + 2 < num_classes:
            mat[i, i + 2] = np.log(0.009) # skip a phase (rare)
        if i - 1 >= 0:
            mat[i, i - 1] = np.log(0.001) # go back one phase (very rare)

    return mat


def compute_transition_matrix(labels_dir: str, num_classes: int,
                               samples: list) -> np.ndarray:
    """
    Estimate transition matrix from training data instead of using defaults.
    samples: list of {labels: path} dicts from training set only.
    """
    counts = np.zeros((num_classes, num_classes))
    for s in samples:
        labels = torch.load(s["labels"]).tolist()
        valid = [l for l in labels if l != -100]
        for i in range(len(valid) - 1):
            counts[valid[i], valid[i + 1]] += 1

    # normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums
    return np.log(probs + 1e-10)
