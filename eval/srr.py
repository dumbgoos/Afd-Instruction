import re
import numpy as np
import pandas as pd


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
AA20_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_seq(s):
    """Clean sequence: uppercase, remove whitespace, keep only 20 standard amino acids"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = re.sub(r"\s+", "", str(s)).upper()
    s = AA20_RE.sub("", s)
    return s


def srr_truncate(a, b, align="left"):
    """Calculate SRR by truncating to shortest length (left or right alignment)"""
    a = clean_seq(a)
    b = clean_seq(b)
    L = min(len(a), len(b))
    if L == 0:
        return np.nan
    if align == "left":
        a_, b_ = a[:L], b[:L]
    elif align == "right":
        a_, b_ = a[-L:], b[-L:]
    else:
        raise ValueError("align must be 'left' or 'right'")
    matches = sum(x == y for x, y in zip(a_, b_))
    return matches / L


def srr_best_overlap(a, b):
    """Calculate best overlap SRR by sliding shorter sequence over longer sequence"""
    a = clean_seq(a)
    b = clean_seq(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    
    if len(a) <= len(b):
        short, long = a, b
    else:
        short, long = b, a
    
    Ls, Ll = len(short), len(long)
    best = 0.0
    for start in range(0, Ll - Ls + 1):
        seg = long[start:start+Ls]
        matches = sum(x == y for x, y in zip(short, seg))
        best = max(best, matches / Ls)
    return best


def calculate_srr_metrics(gt_sequences, pred_sequences):
    """
    Calculate SRR metrics for ground truth and predicted sequences.
    
    Args:
        gt_sequences: List of ground truth sequences
        pred_sequences: List of predicted sequences
    
    Returns:
        dict: Dictionary containing SRR metrics and summary statistics
    """
    if len(gt_sequences) != len(pred_sequences):
        raise ValueError("Ground truth and prediction sequences must have same length")
    
    # Clean sequences
    gt_clean = [clean_seq(seq) for seq in gt_sequences]
    pred_clean = [clean_seq(seq) for seq in pred_sequences]
    
    # Calculate lengths
    len_gt = [len(seq) for seq in gt_clean]
    len_pred = [len(seq) for seq in pred_clean]
    len_min = [min(g, p) for g, p in zip(len_gt, len_pred)]
    len_diff = [p - g for g, p in zip(len_gt, len_pred)]
    
    # Calculate SRR metrics
    srr_left = [srr_truncate(g, p, "left") for g, p in zip(gt_clean, pred_clean)]
    srr_right = [srr_truncate(g, p, "right") for g, p in zip(gt_clean, pred_clean)]
    srr_best = [srr_best_overlap(g, p) for g, p in zip(gt_clean, pred_clean)]
    
    # Exact match (same length and sequence)
    exact_match = [(g == p and len_gt[i] == len_pred[i]) for i, (g, p) in enumerate(zip(gt_clean, pred_clean))]
    
    # Calculate summary statistics for valid sequences
    valid_indices = [i for i, l in enumerate(len_min) if l > 0]
    
    if len(valid_indices) == 0:
        return {
            "N_total": len(gt_sequences),
            "N_valid": 0,
            "exact_match_pct": 0.0,
            "srr_left_mean": np.nan,
            "srr_right_mean": np.nan,
            "srr_best_mean": np.nan,
            "len_gt_mean": np.nan,
            "len_pred_mean": np.nan
        }
    
    valid_exact = [exact_match[i] for i in valid_indices]
    valid_srr_left = [srr_left[i] for i in valid_indices if not np.isnan(srr_left[i])]
    valid_srr_right = [srr_right[i] for i in valid_indices if not np.isnan(srr_right[i])]
    valid_srr_best = [srr_best[i] for i in valid_indices if not np.isnan(srr_best[i])]
    valid_len_gt = [len_gt[i] for i in valid_indices]
    valid_len_pred = [len_pred[i] for i in valid_indices]
    
    return {
        "N_total": len(gt_sequences),
        "N_valid": len(valid_indices),
        "exact_match_pct": np.mean(valid_exact) * 100 if valid_exact else 0.0,
        "srr_left_mean": np.mean(valid_srr_left) if valid_srr_left else np.nan,
        "srr_right_mean": np.mean(valid_srr_right) if valid_srr_right else np.nan,
        "srr_best_mean": np.mean(valid_srr_best) if valid_srr_best else np.nan,
        "len_gt_mean": np.mean(valid_len_gt) if valid_len_gt else np.nan,
        "len_pred_mean": np.mean(valid_len_pred) if valid_len_pred else np.nan
    }


def calculate_srr_detailed(gt_sequences, pred_sequences):
    """
    Calculate detailed SRR metrics for each sequence pair.
    
    Args:
        gt_sequences: List of ground truth sequences
        pred_sequences: List of predicted sequences
    
    Returns:
        dict: Dictionary containing detailed metrics for each sequence
    """
    if len(gt_sequences) != len(pred_sequences):
        raise ValueError("Ground truth and prediction sequences must have same length")
    
    # Clean sequences
    gt_clean = [clean_seq(seq) for seq in gt_sequences]
    pred_clean = [clean_seq(seq) for seq in pred_sequences]
    
    # Calculate all metrics
    results = {
        "gt_clean": gt_clean,
        "pred_clean": pred_clean,
        "len_gt": [len(seq) for seq in gt_clean],
        "len_pred": [len(seq) for seq in pred_clean],
        "len_min": [min(len(g), len(p)) for g, p in zip(gt_clean, pred_clean)],
        "len_diff": [len(p) - len(g) for g, p in zip(gt_clean, pred_clean)],
        "srr_left": [srr_truncate(g, p, "left") for g, p in zip(gt_clean, pred_clean)],
        "srr_right": [srr_truncate(g, p, "right") for g, p in zip(gt_clean, pred_clean)],
        "srr_best": [srr_best_overlap(g, p) for g, p in zip(gt_clean, pred_clean)],
        "exact_match": [(g == p and len(g) == len(p)) for g, p in zip(gt_clean, pred_clean)]
    }
    
    return results
