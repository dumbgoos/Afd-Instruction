def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def exact_match(preds, refs) -> float:
    hits = sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r))
    return hits / len(refs) if refs else 0.0
