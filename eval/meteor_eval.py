import evaluate


meteor_metric = evaluate.load("./meteor.py")


def avg_meteor(refs, preds) -> float:
    results = meteor_metric.compute(predictions=preds, references=refs)
    return results["meteor"] * 100
