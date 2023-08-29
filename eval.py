import argparse
import json
import logging
import os

import numpy as np
from PIL import Image
from tabulate import tabulate

from evaluators import CLIPScoreEvaluator

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_NAME_TO_EVALUATOR = {
    "clip_score": {
        "evaluator": CLIPScoreEvaluator,
        "description ": "This metrics corresponds to the cosine similarity between the visual CLIP embedding for "
                        "an image and the textual CLIP embedding for a caption. The score is bound between 0 and "
                        "100 with 100 being the best score. For more info, check out https://arxiv.org/abs/2104.08718"
    }
}


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m", help="what metrics to evaluate as comma-delimited str",
                        type=str,
                        required=True)
    parser.add_argument("--images", "-i", help="path to directory containing generated images to evaluate",
                        type=str,
                        required=True)
    parser.add_argument("--prompts", "-p", help="path to file containing mapping from image to associated prompt",
                        type=str,
                        required=True)
    parser.add_argument("--available-metrics",
                        help="description of all the available metrics and a synopsis of their properties",
                        action="store_true")
    return parser.parse_args()


def main():
    args = read_args()
    if args.available_metrics:
        metric_descr = []
        for metric, info in METRIC_NAME_TO_EVALUATOR.items():
            metric_descr.append([metric.replace("_", " "), info["description"]])
        tabulate(metric_descr, headers=["Metric Name", "Description"], tablefmt="orgtbl")

    # Get mapping from images to prompts
    with open(args.prompts) as f:
        data = json.load(f)

    # Ingest images
    images = []
    for image in data.keys():
        full_image_path = os.path.join(args.images, image)
        pil_image = Image.open(full_image_path).convert("RGB")
        np_arr = np.asarray(pil_image)
        # Transpose the image arr so that channel is first dim
        images.append(np_arr.transpose(2, 0, 1))

    # Parse str list of metrics
    metrics = args.metrics.split(",")
    computed_metrics = []
    # Compute all metrics
    for metric in metrics:
        try:
            metric_evaluator = METRIC_NAME_TO_EVALUATOR[metric]["evaluator"]
        except:
            logging.error(f"Provided metric {metric} does not exist")
            continue
        computed_metric = metric_evaluator().evaluate(images, list(data.values()))
        computed_metrics.append([metric.replace("_", " "), computed_metric.item()])

    # Print all results
    print(tabulate(computed_metrics, headers=["Metric Name", "Value"], tablefmt="orgtbl"))


if __name__ == "__main__":
    main()
