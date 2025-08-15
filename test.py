def create_metric_fn(metric_option: str = "combined", weights: tuple[float, float, float] = (0.4, 0.3, 0.3)):
    """
    Returns a metric function that computes one of several possible metrics.

    Args:
        metric_option: One of {"combined", "accuracy", "confidence", "evidence"}.
                       "combined" uses the weighted combination of the three scores.
                       "accuracy" returns only the boolean match on `is_compelling`.
                       "confidence" returns only the confidence closeness.
                       "evidence" returns only the evidence‑similarity F1.
        weights:  A 3‑tuple of weights (acc_weight, conf_weight, evidence_weight) used
                  when metric_option == "combined".

    Returns:
        A function metric_fn(example, pred, *args) that can be passed to DSPy.
    """

    # Normalize weights just in case
    w_acc, w_conf, w_f1 = weights
    total = w_acc + w_conf + w_f1
    if total > 0:
        w_acc /= total
        w_conf /= total
        w_f1 /= total

    def metric_fn(example, pred, *args):
        # Boolean accuracy
        gold_bool = parse_bool(example.is_compelling)
        pred_bool = parse_bool(getattr(pred, "is_compelling", ""))
        acc = 1.0 if gold_bool == pred_bool else 0.0

        # Confidence closeness
        gold_c = parse_float(example.confidence_score)
        pred_c = parse_float(getattr(pred, "confidence_score", "0"))
        pred_c = max(0.0, min(1.0, pred_c))      # clamp between 0 and 1
        conf_score = 1.0 - min(1.0, abs(gold_c - pred_c))

        # Evidence similarity (F1 on word overlap)
        gold_words = set(str(example.evidence_details).lower().split())
        pred_words = set(str(getattr(pred, "evidence_details", "")).lower().split())
        intersection = len(gold_words & pred_words)
        prec = intersection / (len(pred_words) + 1e-9)
        rec  = intersection / (len(gold_words) + 1e-9)
        evidence_f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

        # Return the requested metric
        if metric_option == "accuracy":
            return acc
        elif metric_option == "confidence":
            return conf_score
        elif metric_option == "evidence":
            return evidence_f1
        else:  # "combined" or any other value defaults to the weighted sum
            return w_acc * acc + w_conf * conf_score + w_f1 * evidence_f1

    return metric_fn
