import sacrebleu


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def corpus_bleu(preds, refs, n: int = 4) -> float:
    preds_n = [normalize_text(p) for p in preds]
    refs_n = [[normalize_text(r) for r in refs]]
    bleu = sacrebleu.metrics.BLEU(
        smooth_method="exp",
        effective_order=True,
        max_ngram_order=n
    )
    return bleu.corpus_score(preds_n, refs_n).score
