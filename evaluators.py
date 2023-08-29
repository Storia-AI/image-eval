import abc

from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
import torch

torch.manual_seed(42)

class BaseReferenceFreeEvaluator(abc.ABC):
    """
    An evaluation that doesn't require gold samples to compare against.
    """

    @abc.abstractmethod
    def evaluate(self, images: list[Image], prompts: list[str]):
        pass


class CLIPScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self):
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def evaluate(self, images: list[Image], prompts: list[str]):
        pass

