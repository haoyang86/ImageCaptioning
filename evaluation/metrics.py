from typing import List, Dict

from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction,
)

from nltk.translate.meteor_score import meteor_score

from rouge_score import rouge_scorer

from pycocoevalcap.cider.cider import Cider


# ---------------------------------------------------
# BLEU
# ---------------------------------------------------

def compute_bleu_scores(
    references: List[List[str]],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU-1 ... BLEU-4
    """

    refs_tokenized = [
        [ref.split() for ref in refs]
        for refs in references
    ]

    preds_tokenized = [
        pred.split()
        for pred in predictions
    ]

    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(1,0,0,0),
        smoothing_function=smooth,
    )

    bleu2 = corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(0.5,0.5,0,0),
        smoothing_function=smooth,
    )

    bleu3 = corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(1/3,1/3,1/3,0),
        smoothing_function=smooth,
    )

    bleu4 = corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(0.25,0.25,0.25,0.25),
        smoothing_function=smooth,
    )

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
    }


# ---------------------------------------------------
# METEOR
# ---------------------------------------------------

def compute_meteor(
    references: List[List[str]],
    predictions: List[str],
) -> float:

    scores = []

    for refs, pred in zip(
        references,
        predictions
    ):
        score = meteor_score(
            [r.split() for r in refs],
            pred.split(),
        )

        scores.append(score)

    return sum(scores)/len(scores)


# ---------------------------------------------------
# ROUGE-L
# ---------------------------------------------------

def compute_rouge_l(
    references: List[List[str]],
    predictions: List[str],
) -> float:
    """
    Uses best matching reference
    """

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = []

    for refs, pred in zip(
        references,
        predictions
    ):

        best_f = 0.0

        for ref in refs:

            result = scorer.score(
                ref,
                pred
            )

            fscore = result[
                "rougeL"
            ].fmeasure

            if fscore > best_f:
                best_f = fscore

        scores.append(best_f)

    return sum(scores)/len(scores)


# ---------------------------------------------------
# CIDEr
# ---------------------------------------------------

def compute_cider(
    references: List[List[str]],
    predictions: List[str],
) -> float:
    """
    Compute CIDEr using pycocoevalcap

    Required format:

    gts:
       {
         "0":[ref1, ref2, ...]
       }

    res:
       {
         "0":[prediction]
       }
    """

    gts = {}
    res = {}

    for idx, (refs, pred) in enumerate(
        zip(
            references,
            predictions
        )
    ):
        image_id = str(idx)

        gts[image_id] = refs
        res[image_id] = [pred]

    scorer = Cider()

    score, _ = scorer.compute_score(
        gts,
        res
    )

    return float(score)


# ---------------------------------------------------
# All Metrics
# ---------------------------------------------------

def compute_all_metrics(
    references: List[List[str]],
    predictions: List[str],
) -> Dict:

    results = {}

    results.update(
        compute_bleu_scores(
            references,
            predictions
        )
    )

    results["METEOR"] = compute_meteor(
        references,
        predictions
    )

    results["ROUGE-L"] = compute_rouge_l(
        references,
        predictions
    )

    results["CIDEr"] = compute_cider(
        references,
        predictions
    )

    return results


# ---------------------------------------------------
# Pretty print
# ---------------------------------------------------

def print_metrics(
    metrics_dict: Dict
):

    print("\nEvaluation Metrics")
    print("-"*40)

    for k,v in metrics_dict.items():

        print(
            f"{k}: {v:.4f}"
        )