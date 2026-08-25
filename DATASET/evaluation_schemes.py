"""
evaluation_schemes.py
=====================
Extended evaluation for multilabel emotion classification (EmoNoBa).

Scheme 1 – Existing (already in baseline):
    Per-label Macro/Micro/Weighted F1 + Hamming loss (sklearn).

Scheme 2 – Exact Match (new):
    A sample scores 1 only when its entire label vector is identical to gold.
    Precision = Recall = F1 = Exact-Match Ratio (EMR) because there is no
    partial credit; the three coincide for this strict measure.

Scheme 3 – Partial Match / Instance-level (new):
    Per-instance Precision, Recall, and F1 are computed over the *set* of
    positive labels, then averaged (macro-instance). This gives partial credit
    when the prediction overlaps with gold without being identical.
"""

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, classification_report


# ──────────────────────────────────────────────────────────────────────────────
# Scheme 2 : Exact Match
# ──────────────────────────────────────────────────────────────────────────────

def exact_match_scores(gold: np.ndarray, preds: np.ndarray) -> dict:
    """
    Exact-label-match evaluation for multilabel classification.

    A sample is a match only when *every* label in the predicted vector equals
    the corresponding gold label (equivalent to sklearn's subset_accuracy /
    accuracy_score on multilabel arrays).

    Because there is no partial credit, Precision = Recall = F1 = EMR.

    Parameters
    ----------
    gold  : np.ndarray, shape (N, L), dtype int  – ground-truth binary matrix
    preds : np.ndarray, shape (N, L), dtype int  – predicted binary matrix

    Returns
    -------
    dict with keys: precision, recall, f1, exact_match_ratio, num_exact_matches
    """
    assert gold.shape == preds.shape, "gold and preds must have the same shape"

    # Row-wise equality: True only when ALL labels in a row match
    row_matches = np.all(gold == preds, axis=1)          # (N,) bool
    emr = float(row_matches.mean())                      # Exact Match Ratio

    # For a strict binary decision (match vs. no-match):
    #   TP = matched samples, FP = 0, FN = 0  →  P = R = F1 = EMR
    return {
        "precision"         : emr,
        "recall"            : emr,
        "f1"                : emr,
        "exact_match_ratio" : emr,
        "num_exact_matches" : int(row_matches.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scheme 3 : Partial Match  (instance-level set-based P / R / F1)
# ──────────────────────────────────────────────────────────────────────────────

def _instance_prf(gold_row: np.ndarray, pred_row: np.ndarray):
    """
    Compute per-instance Precision, Recall, F1 treating each row as a *set*
    of active labels.

    Edge cases
    ----------
    - If both gold and pred are all-zero  →  perfect match  (P=R=F1=1)
    - If only one is all-zero             →  no overlap     (P=R=F1=0)
    """
    gold_set = set(np.where(gold_row == 1)[0])
    pred_set = set(np.where(pred_row == 1)[0])

    # Both empty → perfect prediction
    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0

    # One empty, the other not → zero overlap
    if len(gold_set) == 0 or len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    overlap = len(gold_set & pred_set)
    precision = overlap / len(pred_set)
    recall    = overlap / len(gold_set)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def partial_match_scores(gold: np.ndarray, preds: np.ndarray) -> dict:
    """
    Instance-level (partial-match) Precision, Recall, and F1.

    For each sample, the positive label sets of gold and prediction are
    intersected and per-instance P/R/F1 values are averaged (macro over
    instances).

    Parameters
    ----------
    gold  : np.ndarray, shape (N, L), dtype int
    preds : np.ndarray, shape (N, L), dtype int

    Returns
    -------
    dict with keys: precision, recall, f1  (all macro-averaged over samples)
    """
    assert gold.shape == preds.shape

    precisions, recalls, f1s = [], [], []
    for g_row, p_row in zip(gold, preds):
        p, r, f = _instance_prf(g_row, p_row)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "precision" : float(np.mean(precisions)),
        "recall"    : float(np.mean(recalls)),
        "f1"        : float(np.mean(f1s)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Combined evaluation function  (drop-in replacement / extension of `evaluate`)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all_schemes(
    gold: np.ndarray,
    preds: np.ndarray,
    label_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run all three evaluation schemes and return a unified results dict.

    Parameters
    ----------
    gold        : np.ndarray (N, L) int  – ground-truth
    preds       : np.ndarray (N, L) int  – predicted (binary, after threshold)
    label_names : list[str] | None       – optional label names for the report
    verbose     : bool                   – print a formatted summary

    Returns
    -------
    dict with nested keys:
        "scheme1" : per-label sklearn metrics
        "scheme2" : exact match metrics
        "scheme3" : partial match metrics
    """

    # ── Scheme 1 : Per-label sklearn metrics ──────────────────────────────────
    scheme1 = {
        "macro_f1"    : f1_score(gold, preds, average="macro",    zero_division=0),
        "micro_f1"    : f1_score(gold, preds, average="micro",    zero_division=0),
        "weighted_f1" : f1_score(gold, preds, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(gold, preds),
    }

    # ── Scheme 2 : Exact match ────────────────────────────────────────────────
    scheme2 = exact_match_scores(gold, preds)

    # ── Scheme 3 : Partial match ──────────────────────────────────────────────
    scheme3 = partial_match_scores(gold, preds)

    results = {"scheme1": scheme1, "scheme2": scheme2, "scheme3": scheme3}

    if verbose:
        _print_report(results, gold, preds, label_names)

    return results


def _print_report(results, gold, preds, label_names):
    sep = "=" * 62

    print(sep)
    print("  SCHEME 1 – Per-label sklearn metrics  (existing baseline)")
    print(sep)
    s1 = results["scheme1"]
    print(f"  Macro  F1      : {s1['macro_f1']:.4f}")
    print(f"  Micro  F1      : {s1['micro_f1']:.4f}")
    print(f"  Weighted F1    : {s1['weighted_f1']:.4f}")
    print(f"  Hamming Loss   : {s1['hamming_loss']:.4f}")
    if label_names:
        print("\n  Per-label classification report:")
        print(
            classification_report(
                gold, preds,
                target_names=label_names,
                digits=4,
                zero_division=0,
            )
        )

    print(sep)
    print("  SCHEME 2 – Exact Match")
    print(sep)
    s2 = results["scheme2"]
    print(f"  Exact Match Ratio (EMR) : {s2['exact_match_ratio']:.4f}")
    print(f"  Num Exact Matches       : {s2['num_exact_matches']}")
    print(f"  Precision               : {s2['precision']:.4f}")
    print(f"  Recall                  : {s2['recall']:.4f}")
    print(f"  F1                      : {s2['f1']:.4f}")
    print("  [Note: P = R = F1 = EMR for strict exact-match scoring]")

    print(sep)
    print("  SCHEME 3 – Partial Match  (instance-level set overlap)")
    print(sep)
    s3 = results["scheme3"]
    print(f"  Precision (macro-inst.) : {s3['precision']:.4f}")
    print(f"  Recall    (macro-inst.) : {s3['recall']:.4f}")
    print(f"  F1        (macro-inst.) : {s3['f1']:.4f}")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Integration with the existing `evaluate()` loop
# ──────────────────────────────────────────────────────────────────────────────

"""
Replace (or extend) the existing evaluate() function in the notebook like so:

    from evaluation_schemes import evaluate_all_schemes

    # After collecting all_preds and all_gold from the eval loop:
    results = evaluate_all_schemes(
        gold        = all_gold,
        preds       = all_preds,
        label_names = EMOTION_COLS,
        verbose     = True,
    )

    # Access individual metrics:
    emr      = results["scheme2"]["exact_match_ratio"]
    partial_f1 = results["scheme3"]["f1"]
"""


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check (runs when executed directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    EMOTION_COLS = ["Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]

    # Simulate gold labels and predictions
    gold  = (rng.random((50, 6)) > 0.6).astype(int)
    preds = gold.copy()

    # Flip ~15 % of bits to simulate imperfect predictions
    flip_mask = rng.random((50, 6)) < 0.15
    preds[flip_mask] = 1 - preds[flip_mask]

    # --- Manual examples from the task description ---
    print("Manual example checks")
    print("-" * 40)

    g1 = np.array([[0, 0, 1, 1, 0, 1]])
    p1 = np.array([[0, 0, 1, 1, 0, 1]])   # perfect
    g2 = np.array([[0, 0, 1, 1, 0, 1]])
    p2 = np.array([[0, 0, 1, 1, 0, 0]])   # one label off

    print("Example 1 (perfect):")
    print(f"  Exact match score  : {exact_match_scores(g1, p1)['exact_match_ratio']}")
    print(f"  Partial match F1   : {partial_match_scores(g1, p1)['f1']:.4f}")

    print("Example 2 (one label off):")
    print(f"  Exact match score  : {exact_match_scores(g2, p2)['exact_match_ratio']}")
    print(f"  Partial match F1   : {partial_match_scores(g2, p2)['f1']:.4f}")

    print("\nFull 50-sample evaluation:")
    evaluate_all_schemes(gold, preds, label_names=EMOTION_COLS, verbose=True)