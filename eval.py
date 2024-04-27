import argparse
import json
import logging
import os
import sys

import torch
from PIL import Image
from tabulate import tabulate

from image_eval.evaluators import (
    AestheticPredictorEvaluator,
    CLIPScoreEvaluator,
    CMMDEvaluator,
    EvaluatorType,
    FIDEvaluator,
    HumanPreferenceScoreEvaluator,
    ImageRewardEvaluator,
    InceptionScoreEvaluator,
    CentroidSimilarityEvaluator,
    VendiScoreEvaluator
)
from image_eval.pairwise_evaluators import (
    LPIPSEvaluator,
    MultiSSIMEvaluator,
    PSNREvaluator,
    UIQIEvaluator
)

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
    "centroid_similarity": {
        "evaluator": CentroidSimilarityEvaluator,
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
    },
    "lpips_score": {
        "evaluator": LPIPSEvaluator,
        "description": "Calculates Learned Perceptual Image Patch Similarity (LPIPS) score as the distance between the "
        "activations of two image patches for some deep network (VGG). The score is between 0 and 1, where 0 means the "
        "images are identical. See https://arxiv.org/pdf/1801.03924.",
    },
    "multi_ssime_score": {
        "evaluator": MultiSSIMEvaluator,
        "description": "Calculates Multi-scale Structural Similarity Index Measure (SSIM). This is an extension of "
        "SSIM, which assesses the similarity between two images based on three components: luminance, contrast, and "
        "structure. The score is between -1 and 1, where 1 = perfect similarity, 0 = no similarity and "
        "-1 = perfect anti-corelation. See https://ieeexplore.ieee.org/document/1292216.",
    },
    "psnr_score": {
        "evaluator": PSNREvaluator,
        "description": "Calculates Peak Signal-to-Noise Ratio (PSNR). It was originally designed to measure the "
        "quality of reconstructed or compressed images compared to their original versions. Its values are between "
        "-infinity and +infinity, where identical images score +infinity. See "
        "https://ieeexplore.ieee.org/document/1163711.",
    },
    "uiqi_score": {
        "evaluator": UIQIEvaluator,
        "description": "Calculates Universal Image Quality Index (UIQI). Based on the idea of comparing statistical "
        "properties of an original and a distorted image in both the spatial and frequency domains. The calculation "
        "involves several steps, including the computation of mean, variance, and covariance of the pixel values in "
        "local windows of the images. It also considers factors like luminance, contrast, and structure. See "
        "https://ieeexplore.ieee.org/document/1284395.",
    },
}


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m",
                        default="all",
                        help="valid values are: (1) all, (2) one of the following categories: image_quality, "
                             "controllability, fidelity, pairwise_similarity, diversity, or (3) a comma-separated list "
                             "of metric names, for example: clip_score,style_similarity.",
                        type=str)
    parser.add_argument("--generated-images", "-g",
                        help="path to directory containing generated images to evaluate; for pairwise_similarity, "
                             "-g and -r need to have the same number of images with the same filenames.",
                        type=str,
                        required=True)
    parser.add_argument("--real-images", "-r",
                        help="path to directory containing real images to use for evaluation; for pairwise_similarity, "
                             "-g and -r need to have the same number of images with the same filenames.",
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
                        default=AestheticPredictorEvaluator.DEFAULT_URL,
                        help="Model checkpoint for the aesthetic predictor evaluator.")
    parser.add_argument("--human-preference-score-model-url",
                        default=HumanPreferenceScoreEvaluator.DEFAULT_URL,
                        help="Model checkpoint for the human preference score evaluator.")
    return parser.parse_args()


def read_images(image_dir: str) -> Dict[str, Image.Image]:
    """Reads all the images in a given folder."""
    images = {}
    # It's important to sort the filenames for pairwise similarity metrics.
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
        # It's important to sort the filenames for pairwise similarity metrics.
        prompts_by_image_filename = sorted(prompts_by_image_filename.items(), key=lambda x: x[0])

    for image_filename, prompt in prompts_by_image_filename:
        image = images_by_filename.get(image_filename)
        if image:
            images.append(image)
            prompts.append(prompt)
        else:
            logging.warning(f"Could not find image {image_filename}. "
                            f"Available images are: {images_by_filename.keys()}")
    return images, prompts


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

    metrics_explicitly_specified = []
    if args.metrics == "all":
        # We exclude PAIRWISE_SIMILARITY metrics from here and only calculate them when explicilty
        # specified by the user, as they require aligned datasets.
        metrics = [name for name, metric in METRIC_NAME_TO_EVALUATOR.items()
                   if metric["evaluator"].TYPE != EvaluatorType.PAIRWISE_SIMILARITY]
        args.metrics = ",".join(metrics)
    elif args.metrics in [member.name.lower() for member in EvaluatorType]:
        args.metrics = ",".join([
            name for name, metric in METRIC_NAME_TO_EVALUATOR.items()
            if metric["evaluator"].TYPE.name.lower() == args.metrics
        ])
    else:
        # Metrics must be comma-separated
        metrics_explicitly_specified = args.metrics.split(",")
    metrics = args.metrics.split(",")

    generated_images_by_filename = read_images(args.generated_images)
    real_images = list(read_images(args.real_images).values()) if args.real_images else None

    if args.prompts:
        generated_images, prompts = read_prompts_for_images(generated_images_by_filename, args.prompts)
    else:
        generated_images = list(generated_images_by_filename.values())
        prompts = None

    computed_metrics = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running evaluation on device: {device}")

    # Compute all metrics
    all_computed_metrics = {}
    for metric in metrics:
        if metric == "aesthetic_predictor":
            evaluator = AestheticPredictorEvaluator(device, args.aesthetic_predictor_model_url)
        elif metric == "human_preference_score":
            evaluator = HumanPreferenceScoreEvaluator(device, args.human_preference_score_model_url)
        else:
            metric_evaluator = METRIC_NAME_TO_EVALUATOR[metric]["evaluator"]
            evaluator = metric_evaluator(device)

        if (not evaluator.should_trigger_for_data(generated_images) and
                not metric in metrics_explicitly_specified):
            logging.warning(f"Skipping metric {metric} as it is not useful for the given images.")
            continue

        logging.info(f"Computing metric {metric}...")
        computed_metrics = evaluator.evaluate(generated_images,
                                              real_images=real_images,
                                              prompts=prompts)
        all_computed_metrics.update(computed_metrics)

    # Print all results
    print(tabulate(all_computed_metrics.items(),
                   headers=["Metric Name", "Value"],
                   tablefmt="orgtbl"))


if __name__ == "__main__":
    # If we don't explicitly mark all models for inference, Huggingface seems to hold on to some
    # object references even after they're not needed anymore (perhaps to keep gradients around),
    # which causes this script to OOM when multiple evaluators are run in a sequence.
    # See https://github.com/huggingface/transformers/issues/26275.
    with torch.inference_mode():
        main()
