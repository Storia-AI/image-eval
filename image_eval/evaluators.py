import abc
import os
from typing import Union

import ImageReward as RM
import PIL
import clip
import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

from image_eval.improved_aesthetic_predictor import run_inference

torch.manual_seed(42)


# TODO (mihail): Decouple this so not all evaluators are in the same file

class BaseReferenceFreeEvaluator(abc.ABC):
    """
    An evaluation that doesn't require gold samples to compare against.
    """
    def __init__(self, device: str):
        self.device = device

    @abc.abstractmethod
    def evaluate(self, images: list[Union[np.array, Image.Image]], prompts: list[str]):
        pass


class BaseWithReferenceEvaluator(abc.ABC):
    """
    An evaluation that includes gold samples to compare against.
    """
    def __init__(self, device: str):
        self.device = device

    @abc.abstractmethod
    def evaluate(self, generated_images: list[np.array], real_images: list[np.array]):
        pass


class CLIPScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)

    def evaluate(self, images: list[np.array], prompts: list[str]):
        torch_imgs = [torch.tensor(img).to(self.device) for img in images]
        self.evaluator.update(torch_imgs, prompts)
        return self.evaluator.compute()


class InceptionScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = InceptionScore().to(self.device)

    def evaluate(self, images: list[np.array], ignored_prompts: list[str]):
        torch_imgs = torch.stack([torch.tensor(img).to(self.device) for img in images])
        self.evaluator.update(torch_imgs)
        return self.evaluator.compute()


class FIDEvaluator(BaseWithReferenceEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator64 = FrechetInceptionDistance(feature=64).to(self.device).set_dtype(torch.float64)
        self.evaluator192 = FrechetInceptionDistance(feature=192).to(self.device).set_dtype(torch.float64)
        self.evaluator768 = FrechetInceptionDistance(feature=768).to(self.device).set_dtype(torch.float64)
        self.evaluator2048 = FrechetInceptionDistance(feature=2048).to(self.device).set_dtype(torch.float64)

    def evaluate(self, generated_images: list[np.array], real_images: list[str]):
        torch_gen_imgs = torch.stack([torch.tensor(img).to(self.device) for img in generated_images])
        torch_real_imgs = torch.stack([torch.tensor(img).to(self.device) for img in real_images])
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
    def __init__(self, device: str):
        super().__init__(device)

    def evaluate(self, images: list[Image.Image], ignored_prompts: list[str]):
        return run_inference(images, self.device)


class ImageRewardEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = RM.load("ImageReward-v1.0")

    def evaluate(self, images: list[Image.Image], prompts: list[str]):
        # Returns the average image reward
        rewards = []
        for image, prompt in zip(images, prompts):
            rewards.append(self.evaluator.score(prompt, image))
        return sum(rewards) / len(rewards)


class HumanPreferenceScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        model, preprocess = clip.load("ViT-L/14", device=self.device)
        model_path = os.path.join(os.environ["MODELS_DIR"], "human_preference_score/hpc.pt")
        if torch.cuda.is_available():
            params = torch.load(model_path)['state_dict']
        else:
            params = torch.load(model_path, map_location=self.device)['state_dict']
        model.load_state_dict(params)
        self.model = model
        self.preprocess = preprocess

    def evaluate(self, images: list[PIL.Image], prompts: list[str]):
        # Returns the average human preference score
        scores = []
        # TODO (mihail): Batch the inputs for faster processing
        for pil_img, prompt in zip(images, prompts):
            image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            text = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                hps = image_features @ text_features.T
                hps = hps.diagonal()
                scores.append(hps.squeeze().tolist())

        return sum(scores) / len(scores)
