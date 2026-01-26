from rouge_score import rouge_scorer


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def avg_rouge_f1(refs, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
    r1 = r2 = rl = 0.0
    n = len(refs)
    for ref, pred in zip(refs, preds):
        ref_n = normalize_text(ref)
        pred_n = normalize_text(pred)
        s = scorer.score(ref_n, pred_n)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    return r1/n*100, r2/n*100, rl/n*100 if n > 0 else (0,0,0)
