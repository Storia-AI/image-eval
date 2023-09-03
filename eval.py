import argparse
import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from tabulate import tabulate

from evaluators import BaseReferenceFreeEvaluator
from evaluators import BaseWithReferenceEvaluator
from evaluators import CLIPScoreEvaluator
from evaluators import FIDEvaluator
from evaluators import InceptionScoreEvaluator

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_NAME_TO_EVALUATOR = {
    "clip_score": {
        "evaluator": CLIPScoreEvaluator,
        "description": "This metrics corresponds to the cosine similarity between the visual CLIP embedding for "
                       "an image and the textual CLIP embedding for a caption. The score is bound between 0 and "
                       "100 with 100 being the best score. For more info, check out https://arxiv.org/abs/2104.08718"
    },
    "inception_score": {
        "evaluator": InceptionScoreEvaluator,
        "description": "This metrics uses the Inception V3 model to compute class probabilities for generated images. "
                       "and then calculates the KL divergence between the marginal distribution of the class "
                       "probabilities and the conditional distribution of the class probabilities given the generated "
                       "images. The score is bound between 1 and the number of classes supported by the classification "
                       "model. For more info, check out https://arxiv.org/abs/1606.03498"
    },
    "fid": {
        "evaluator": FIDEvaluator,
        "description": "This metrics uses the Inception V3 model to compute a multivariate gaussian for a set of real images"
                       "as well as a multivariate gaussian for a set of fake images. A distance is then computed using "
                       "the summary statistics of these gaussians. A lower score is better with a 0 being a perfect "
                       "score indicating identical groups of images. This metric computes a distance for features"
                       "derived from the 64, 192, 768, and 2048 feature layers. For more info, check out https://arxiv.org/abs/1512.00567"
    }
}


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m", help="what metrics to evaluate as comma-delimited str",
                        type=str)
    parser.add_argument("--generated-images", "-g", help="path to directory containing generated images to evaluate",
                        type=str)
    parser.add_argument("--real-images", "-r", help="path to directory containing real images to use for evaluation",
                        type=str)
    parser.add_argument("--prompts", "-p", help="path to file containing mapping from image to associated prompt",
                        type=str)
    parser.add_argument("--available-metrics",
                        help="description of all the available metrics and a synopsis of their properties",
                        action="store_true")
    return parser.parse_args()


def get_images_from_dir(dir_path: str):
    images = []
    for image in os.listdir(dir_path):
        full_image_path = os.path.join(dir_path, image)
        pil_image = Image.open(full_image_path).convert("RGB")
        np_arr = np.asarray(pil_image)
        # Transpose the image arr so that channel is first dim
        images.append(np_arr.transpose(2, 0, 1))
    return images


def main():
    args = read_args()
    if args.available_metrics:
        metric_descr = []
        for metric, info in METRIC_NAME_TO_EVALUATOR.items():
            metric_descr.append([metric, info["description"]])
        print(tabulate(metric_descr, headers=["Metric Name", "Description"], tablefmt="grid"))
        return

    generated_images = get_images_from_dir(args.generated_images)

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
        evaluator = metric_evaluator()
        if isinstance(evaluator, BaseReferenceFreeEvaluator):
            # Get mapping from images to prompts
            with open(args.prompts) as f:
                data = json.load(f)
            computed_metric = evaluator.evaluate(generated_images, list(data.values()))
        elif isinstance(evaluator, BaseWithReferenceEvaluator):
            real_images = get_images_from_dir(args.real_images)
            computed_metric = evaluator.evaluate(generated_images, real_images)
        if isinstance(computed_metric, torch.Tensor):
            computed_metrics.append([metric, computed_metric.item()])
        elif isinstance(computed_metric, tuple):
            computed_metrics.append([metric, [metric.item() for metric in computed_metric]])

    # Print all results
    print(tabulate(computed_metrics, headers=["Metric Name", "Value"], tablefmt="orgtbl"))


if __name__ == "__main__":
    main()
