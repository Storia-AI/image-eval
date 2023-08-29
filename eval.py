import argparse
import json
import os

from evaluators import CLIPScoreEvaluator
from PIL import Image

METRIC_NAME_TO_EVALUATOR = {
    "clip": CLIPScoreEvaluator
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
    parser.add_argument("--available-metrics", help="description of all the available metrics and a synopsis of their properties",
                        type=str)
    return parser.parse_args()


def main():
    args = read_args()
    # Parse str list of metrics
    metrics = args.metrics.split(",")

    # Get mapping from images to prompts
    with open(args.prompts) as f:
        data = json.load(f)

    # Ingest images
    images = []
    for image in data.keys():
        full_image_path = os.path.join(args.images, image)
        pil_image = Image.open(full_image_path)
        images.append(pil_image)
        print(pil_image.size)
    print(images)


if __name__ == "__main__":
    main()
