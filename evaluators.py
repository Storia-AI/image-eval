import abc

import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore

torch.manual_seed(42)


class BaseReferenceFreeEvaluator(abc.ABC):
    """
    An evaluation that doesn't require gold samples to compare against.
    """

    @abc.abstractmethod
    def evaluate(self, images: list[np.array], prompts: list[str]):
        pass


class CLIPScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self):
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def evaluate(self, images: list[np.array], prompts: list[str]):
        torch_imgs = [torch.tensor(img) for img in images]
        self.evaluator.update(torch_imgs, prompts)
        return self.evaluator.compute()
