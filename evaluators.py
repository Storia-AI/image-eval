import abc

import ImageReward as RM
import PIL
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

from improved_aesthetic_predictor import run_inference

torch.manual_seed(42)


class BaseReferenceFreeEvaluator(abc.ABC):
    """
    An evaluation that doesn't require gold samples to compare against.
    """

    @abc.abstractmethod
    def evaluate(self, images: list[np.array], prompts: list[str]):
        pass


class BaseWithReferenceEvaluator(abc.ABC):
    """
    An evaluation that includes gold samples to compare against.
    """

    @abc.abstractmethod
    def evaluate(self, generated_images: list[np.array], real_images: list[np.array]):
        pass


class CLIPScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self):
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def evaluate(self, images: list[np.array], prompts: list[str]):
        torch_imgs = [torch.tensor(img) for img in images]
        self.evaluator.update(torch_imgs, prompts)
        return self.evaluator.compute()


class InceptionScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self):
        self.evaluator = InceptionScore()

    def evaluate(self, images: list[np.array], ignored_prompts: list[str]):
        torch_imgs = torch.stack([torch.tensor(img) for img in images])
        self.evaluator.update(torch_imgs)
        return self.evaluator.compute()


class FIDEvaluator(BaseWithReferenceEvaluator):
    def __init__(self):
        self.evaluator64 = FrechetInceptionDistance(feature=64)
        self.evaluator192 = FrechetInceptionDistance(feature=192)
        self.evaluator768 = FrechetInceptionDistance(feature=768)
        self.evaluator2048 = FrechetInceptionDistance(feature=2048)

    def evaluate(self, generated_images: list[np.array], real_images: list[str]):
        torch_gen_imgs = torch.stack([torch.tensor(img) for img in generated_images])
        torch_real_imgs = torch.stack([torch.tensor(img) for img in real_images])
        self.evaluator64.update(torch_gen_imgs, real=False)
        self.evaluator64.update(torch_real_imgs, real=True)
        self.evaluator192.update(torch_gen_imgs, real=False)
        self.evaluator192.update(torch_real_imgs, real=True)
        self.evaluator768.update(torch_gen_imgs, real=False)
        self.evaluator768.update(torch_real_imgs, real=True)
        self.evaluator2048.update(torch_gen_imgs, real=False)
        self.evaluator2048.update(torch_real_imgs, real=True)
        return (self.evaluator64.compute(),
                self.evaluator192.compute(),
                self.evaluator768.compute(),
                self.evaluator2048.compute())


class AestheticPredictorEvaluator(BaseReferenceFreeEvaluator):
    def evaluate(self, images: list[PIL.Image], ignored_prompts: list[str]):
        return run_inference(images)


class ImageRewardEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self):
        self.evaluator = RM.load("ImageReward-v1.0")

    def evaluate(self, images: list[PIL.Image], prompts: list[str]):
        # Returns the average image reward
        rewards = []
        for image, prompt in zip(images, prompts):
            rewards.append(self.evaluator.score(prompt, image))
        return sum(rewards) / len(rewards)