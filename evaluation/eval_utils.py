import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from evaluation.decode_utils import greedy_generate, clean_caption_list
from evaluation.metrics import compute_all_metrics, print_metrics


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    return checkpoint


def collect_references_from_dataframe(df) -> Dict[str, List[str]]:
    """
    Group captions by image name.

    Input dataframe columns:
        image, caption

    Output:
        {
            "xxx.jpg": [
                "caption 1",
                "caption 2",
                ...
            ]
        }
    """

    references = {}

    for image_name, group in df.groupby("image"):
        references[image_name] = group["caption"].tolist()

    return references


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset,
    vocab,
    device: torch.device,
    max_len: int,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate predictions for a dataset.

    Important:
        This assumes dataset returns samples in the same order as dataset.df.

    Output item:
        {
            "image": "...jpg",
            "prediction": "...",
            "references": [...]
        }
    """

    model.eval()

    references_by_image = collect_references_from_dataframe(
        dataset.df
    )

    results = []

    global_idx = 0

    for batch_idx, (images, captions) in enumerate(loader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        predictions = greedy_generate(
            model=model,
            images=images,
            vocab=vocab,
            device=device,
            max_len=max_len,
        )

        predictions = clean_caption_list(predictions)

        batch_size = images.size(0)

        for i in range(batch_size):
            image_name = dataset.images.iloc[global_idx]

            item = {
                "image": image_name,
                "prediction": predictions[i],
                "references": references_by_image[image_name],
            }

            results.append(item)
            global_idx += 1

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Generated predictions for "
                f"{batch_idx + 1}/{len(loader)} batches"
            )

    return results


def compute_metrics_from_predictions(
    prediction_items: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute all metrics from saved prediction items.
    """

    predictions = [
        item["prediction"]
        for item in prediction_items
    ]

    references = [
        item["references"]
        for item in prediction_items
    ]

    metrics = compute_all_metrics(
        references=references,
        predictions=predictions,
    )

    return metrics


def run_evaluation(
    model: torch.nn.Module,
    checkpoint_path: Path,
    loader: DataLoader,
    dataset,
    vocab,
    device: torch.device,
    prediction_path: Path,
    metrics_path: Path,
    max_len: int = 50,
    max_batches: Optional[int] = None,
):
    """
    Full offline evaluation pipeline.

    Steps:
        1. load checkpoint
        2. generate predictions
        3. save predictions
        4. compute metrics
        5. save metrics
    """

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val_loss:", checkpoint.get("val_loss"))

    prediction_items = generate_predictions(
        model=model,
        loader=loader,
        dataset=dataset,
        vocab=vocab,
        device=device,
        max_len=max_len,
        max_batches=max_batches,
    )

    save_json(
        prediction_items,
        prediction_path,
    )

    print("Saved predictions:", prediction_path)

    metrics = compute_metrics_from_predictions(
        prediction_items
    )

    save_json(
        metrics,
        metrics_path,
    )

    print_metrics(metrics)
    print("Saved metrics:", metrics_path)

    return metrics