import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image
from tabulate import tabulate

from evaluators import BaseReferenceFreeEvaluator, AestheticPredictorEvaluator, ImageRewardEvaluator, \
    HumanPreferenceScoreEvaluator
from evaluators import BaseWithReferenceEvaluator
from evaluators import CLIPScoreEvaluator
from evaluators import FIDEvaluator
from evaluators import InceptionScoreEvaluator
from streamlit.web import cli as stcli

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
                       "model. The score is computed on random splits of the data so both a mean and standard deviation "
                       "are reported. For more info, check out https://arxiv.org/abs/1606.03498"
    },
    "fid": {
        "evaluator": FIDEvaluator,
        "description": "This metrics uses the Inception V3 model to compute a multivariate gaussian for a set of real images"
                       "as well as a multivariate gaussian for a set of fake images. A distance is then computed using "
                       "the summary statistics of these Gaussians. A lower score is better with a 0 being a perfect "
                       "score indicating identical groups of images. This metric computes a distance for features"
                       "derived from the 64, 192, 768, and 2048 feature layers. For more info, check out https://arxiv.org/abs/1512.00567"
    },
    "aesthetic_predictor": {
        "evaluator": AestheticPredictorEvaluator,
        "description": "This metrics trains a model to predict an aesthetic score using a multilayer perceptron"
                       "trained from the AVA dataset (http://refbase.cvc.uab.es/files/MMP2012a.pdf) using CLIP input embeddings."
                       "A larger score indicates a better model."
    },
    "image_reward": {
        "evaluator": ImageRewardEvaluator,
        "description": "This metrics trains a model to predict image rewards using a dataset of human preferences for images."
                       "Each reward is intended to output a value sampled from a Gaussian with 0 mean and stddev 1. For more details"
                       "check out https://arxiv.org/pdf/2304.05977.pdf"
    },
    "human_preference_score": {
        "evaluator": HumanPreferenceScoreEvaluator,
        "description": "This metric outputs an estimate of the human preference for an image based on the paper https://tgxs002.github.io/align_sd_web/"
                       "The metric is bound between -100 and 100 with 100 being the best score."
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
    parser.add_argument("--local-human-eval",
                        help="run a local instance of streamlit for human evaluation",
                        action="store_true")
    parser.add_argument("--model-predictions-json",
                        help="path to json file containing model predictions")
    return parser.parse_args()


def get_images_from_dir(dir_path: str, convert_to_arr: bool = True):
    images = []
    for image in os.listdir(dir_path):
        full_image_path = os.path.join(dir_path, image)
        pil_image = Image.open(full_image_path).convert("RGB")
        if convert_to_arr:
            np_arr = np.asarray(pil_image)
            # Transpose the image arr so that channel is first dim
            images.append(np_arr.transpose(2, 0, 1))
        else:
            images.append(pil_image)
    return images


def main():
    args = read_args()
    if args.available_metrics:
        metric_descr = []
        for metric, info in METRIC_NAME_TO_EVALUATOR.items():
            metric_descr.append([metric, info["description"]])
        print(tabulate(metric_descr, headers=["Metric Name", "Description"], tablefmt="grid"))
        return

    if args.local_human_eval:
        assert args.model_predictions_json is not None, "Must provide model predictions json"
        sys.argv = ["streamlit", "run", "local_ab_test.py", "--", "--model-predictions-json", args.model_predictions_json]
        sys.exit(stcli.main())

    generated_images = get_images_from_dir(args.generated_images)

    # Parse str list of metrics
    metrics = args.metrics.split(",")
    computed_metrics = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Compute all metrics
    for metric in metrics:
        try:
            metric_evaluator = METRIC_NAME_TO_EVALUATOR[metric]["evaluator"]
        except:
            logging.error(f"Provided metric {metric} does not exist")
            continue
        evaluator = metric_evaluator(device)
        # TODO (mihail): Figure out whether the input to all evaluators can just be PIL.Image
        if isinstance(evaluator, AestheticPredictorEvaluator) or isinstance(evaluator, ImageRewardEvaluator) \
                or isinstance(evaluator, HumanPreferenceScoreEvaluator):
            with open(args.prompts) as f:
                prompts = json.load(f)
            generated_images = get_images_from_dir(args.generated_images, convert_to_arr=False)
            computed_metric = evaluator.evaluate(generated_images, list(prompts.values()))
        elif isinstance(evaluator, BaseReferenceFreeEvaluator):
            # Get mapping from images to prompts
            with open(args.prompts) as f:
                prompts = json.load(f)
            computed_metric = evaluator.evaluate(generated_images, list(prompts.values()))
        elif isinstance(evaluator, BaseWithReferenceEvaluator):
            real_images = get_images_from_dir(args.real_images)
            computed_metric = evaluator.evaluate(generated_images, real_images)
        if isinstance(computed_metric, torch.Tensor):
            computed_metrics.append([metric, computed_metric.item()])
        elif isinstance(computed_metric, tuple):
            computed_metrics.append([metric, [metric.item() for metric in computed_metric]])
        elif isinstance(computed_metric, float):
            computed_metrics.append([metric, computed_metric])

    # Print all results
    print(tabulate(computed_metrics, headers=["Metric Name", "Value"], tablefmt="orgtbl"))


if __name__ == "__main__":
    main()
