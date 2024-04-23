import argparse
import json
import logging
import os
import requests
import sys

import torch
from PIL import Image
from tabulate import tabulate

from image_eval.evaluators import AestheticPredictorEvaluator
from image_eval.evaluators import BaseReferenceFreeEvaluator
from image_eval.evaluators import BaseWithReferenceEvaluator
from image_eval.evaluators import CLIPScoreEvaluator
from image_eval.evaluators import CMMDEvaluator
from image_eval.evaluators import FIDEvaluator
from image_eval.evaluators import HumanPreferenceScoreEvaluator
from image_eval.evaluators import ImageRewardEvaluator
from image_eval.evaluators import InceptionScoreEvaluator
from image_eval.evaluators import StyleSimilarityEvaluator
from image_eval.evaluators import VendiScoreEvaluator

from streamlit.web import cli as stcli
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_NAME_TO_EVALUATOR = {
    "clip_score": {
        "evaluator": CLIPScoreEvaluator,
        "description": "This metric corresponds to the cosine similarity between the visual CLIP embedding for "
                       "an image and the textual CLIP embedding for a caption. The score is bound between 0 and "
                       "100 with 100 being the best score. For more info, check out https://arxiv.org/abs/2104.08718"
    },
    "style_similarity": {
        "evaluator": StyleSimilarityEvaluator,
        "description": "This metric reflects the average cosine similarity between the cluster center of reference images "
                       "and the generated images. The score is bound between 0 and 1, with 1 being the best score. "
                       "The purpose of the metric is to measure how in-style generated images are, compared to the real ones."
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
        "description": "This metrics uses the Inception V3 model to compute a multivariate Gaussian for a set of real images"
                       "as well as a multivariate gaussian for a set of fake images. A distance is then computed using "
                       "the summary statistics of these Gaussians. A lower score is better with a 0 being a perfect "
                       "score indicating identical groups of images. This metric computes a distance for features"
                       "derived from the 64, 192, 768, and 2048 feature layers. For more info, check out https://arxiv.org/abs/1512.00567"
    },
    "cmmd": {
        "evaluator": CMMDEvaluator,
        "description": "A better FID alternative. See https://arxiv.org/abs/2401.09603.",
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
    },
    "vendi_score": {
        "evaluator": VendiScoreEvaluator,
        "description": "This metric evaluates how diverse the generated image set is. We suggest generating all images with the same prompt."
                       "See https://arxiv.org/abs/2210.02410.",
    }
}


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m",
                        default="all",
                        help="what metrics to evaluate as comma-delimited str; if `all`, we compute all possible metrics given the specified flags",
                        type=str)
    parser.add_argument("--generated-images", "-g",
                        help="path to directory containing generated images to evaluate",
                        type=str,
                        required=True)
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
    parser.add_argument("--aesthetic-predictor-model-url",
                        default="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
                        help="Model checkpoint for the aesthetic predictor evaluator.")
    parser.add_argument("--human-preference-score-model-url",
                        default="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?e=b7rgYW",
                        help="Model checkpoint for the human preference score evaluator.")
    parser.add_argument("--cache_dir",
                        default="/tmp/image-eval",
                        help="Path where to download any auxiliary models and data.")
    return parser.parse_args()


def read_images(image_dir: str) -> Dict[str, Image.Image]:
    """Reads all the images in a given folder."""
    images = {}
    image_filenames = sorted(os.listdir(image_dir))
    for image_filename in image_filenames:
        image_path = os.path.join(image_dir, image_filename)
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception:
            # Ignore non-image files in this folder.
            logging.warning(f"Cannot read image from {image_path}. Skipping.")
            continue
        images[image_filename] = pil_image
    return images


def read_prompts_for_images(images_by_filename: Dict[str, Image.Image], prompts_path: str):
    images = []
    prompts = []
    with open(prompts_path, "r") as f:
        prompts_by_image_filename = json.load(f)
    for image_filename, prompt in prompts_by_image_filename.items():
        image = images_by_filename.get(image_filename)
        if image:
            images.append(image)
            prompts.append(prompt)
        else:
            logging.warning(f"Could not find image {image_filename}. "
                            f"Available images are: {images_by_filename.keys()}")
    return images, prompts


def download_model(url: str, output_path: str):
    if os.path.exists(output_path):
        return

    logging.info(f"Downloading model from {url}...")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url)
    if not response.status_code == 200:
        raise RuntimeError(f"Unable to download model from {url}.")

    with open(output_path, "wb") as f:
        f.write(response.content)


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
        lib_folder = os.path.dirname(os.path.realpath(__file__))
        sys.argv = ["streamlit", "run", f"{lib_folder}/image_eval/local_ab_test.py", "--", "--model-predictions-json", args.model_predictions_json]
        sys.exit(stcli.main())

    if args.metrics == "all":
        args.metrics = ",".join(METRIC_NAME_TO_EVALUATOR.keys())

    generated_images_by_filename = read_images(args.generated_images)
    if args.real_images:
        real_images = list(read_images(args.real_images).values())
    if args.prompts:
        generated_images, prompts = read_prompts_for_images(generated_images_by_filename, args.prompts)
    else:
        generated_images = list(generated_images_by_filename.values())
        prompts = None

    # Parse str list of metrics
    metrics = args.metrics.split(",")

    computed_metrics = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Compute all metrics
    for metric in metrics:
        logging.info(f"Computing metric {metric}...")
        try:
            metric_evaluator = METRIC_NAME_TO_EVALUATOR[metric]["evaluator"]
        except:
            logging.error(f"Provided metric {metric} does not exist")
            continue

        if metric == "aesthetic_predictor":
            model_path = os.path.join(args.cache_dir, "aesthetic_predictor/model.pth")
            download_model(url=args.aesthetic_predictor_model_url, output_path=model_path)
            evaluator = AestheticPredictorEvaluator(device, model_path)
        elif metric == "human_preference_score":
            model_path = os.path.join(args.cache_dir, "human_preference_score/model.pth")
            download_model(url=args.human_preference_score_model_url, output_path=model_path)
            evaluator = HumanPreferenceScoreEvaluator(device, model_path)
        else:
            evaluator = metric_evaluator(device)

        if isinstance(evaluator, BaseReferenceFreeEvaluator):
            computed_metric = evaluator.evaluate(generated_images, prompts)
        else:
            assert isinstance(evaluator, BaseWithReferenceEvaluator)
            computed_metric = evaluator.evaluate(generated_images, real_images)

        for key, value in computed_metric.items():
            if isinstance(value, torch.Tensor):
                computed_metrics.append([key, value.item()])
            elif isinstance(value, tuple):
                computed_metrics.append([key, [metric.item() for metric in value]])
            elif isinstance(value, float):
                computed_metrics.append([key, value])
            else:
                raise RuntimeError(f"Unexpected type for computed metric: {type(computed_metric)}")

    # Print all results
    print(tabulate(computed_metrics, headers=["Metric Name", "Value"], tablefmt="orgtbl"))


if __name__ == "__main__":
    main()
