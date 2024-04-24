"""Compares the quality of two images."""
import argparse
import torch

from PIL import Image
from tabulate import tabulate

from image_eval.evaluators import EvaluatorType
from image_eval.evaluators import get_evaluators_for_type


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", "-i1", help="Path to the first image", type=str, required=True)
    parser.add_argument("--image2", "-i2", help="Path to the second image", type=str, required=True)
    parser.add_argument("--prompt", "-p", help="Prompt used to generated the images", type=str)
    return parser.parse_args()

def main():
    args = read_args()
    image1 = Image.open(args.image1).convert("RGB")
    image2 = Image.open(args.image2).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    individual_metrics = []
    for evaluator_cls in get_evaluators_for_type(EvaluatorType.IMAGE_QUALITY):
        evaluator = evaluator_cls(device)
        if not evaluator.is_useful([image1]):
            continue
        results1 = evaluator.evaluate(generated_images=[image1])
        results2 = evaluator.evaluate(generated_images=[image2])
        for metric, value1 in results1.items():
            value2 = results2[metric]
            individual_metrics.append((metric, value1, value2))

    if args.prompt:
        for evaluator_cls in get_evaluators_for_type(EvaluatorType.CONTROLLABILITY):
            evaluator = evaluator_cls(device)
            if not evaluator.is_useful([image1]):
                continue
            results1 = evaluator.evaluate(generated_images=[image1], prompts=[args.prompt])
            results2 = evaluator.evaluate(generated_images=[image2], prompts=[args.prompt])
            for metric, value1 in results1.items():
                value2 = results2[metric]
                individual_metrics.append((metric, value1, value2))

    print(tabulate(individual_metrics,
                   headers=["Metric Name", "Image 1", "Image 2"],
                   tablefmt="orgtbl"))

    fidelity_metrics = []
    for evaluator_cls in get_evaluators_for_type(EvaluatorType.FIDELITY):
        evaluator = evaluator_cls(device)
        if not evaluator.is_useful([image1]):
            continue
        results = evaluator.evaluate(generated_images=[image1], real_images=[image2])
        fidelity_metrics.extend([(metric, value) for metric, value in results.items()])

    print(tabulate(fidelity_metrics,
                   headers=["Metric Name", "Fidelity Score"],
                   tablefmt="orgtbl"))


if __name__ == "__main__":
    # https://github.com/huggingface/transformers/issues/26275
    with torch.inference_mode():
        main()
