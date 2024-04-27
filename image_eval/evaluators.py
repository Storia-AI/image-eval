import abc

import ImageReward as RM
import clip
import torch
from PIL import Image
from enum import Enum
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from typing import Dict
from vendi_score import vendi

from image_eval.improved_aesthetic_predictor import run_inference
from image_eval.encoders import ALL_ENCODER_CLASSES
from image_eval.model_utils import download_model

torch.manual_seed(42)


# TODO (mihail): Decouple this so not all evaluators are in the same file


class EvaluatorType(Enum):
    """Follows the evaluation terminology established by https://arxiv.org/pdf/2309.14859.pdf."""
    # The visual appeal of the generated images (naturalness, absence of artifacts or deformations).
    IMAGE_QUALITY = 1

    # The model’s ability to generate images that align well with text prompts.
    CONTROLLABILITY = 2

    # The extent to which generated images adhere to the target concept. Relevant for fine-tuned
    # models rather than foundational ones (e.g. vanilla Stable Diffusion).
    FIDELITY = 3

    # The pairwise similarity between two sets of images. While FIDELITY allows the two sets to have
    # different sizes and makes bulk comparisons, PAIRWISE_SIMILARITY requires a 1:1 correspondence
    # between images and makes pairwise comparisons.
    PAIRWISE_SIMILARITY = 4

    # The variety of images that are produced from a single or a set of prompts.
    DIVERSITY = 5


class BaseEvaluator(abc.ABC):
    """Base class for evaluators."""
    def __init__(self, device: str):
        self.device = device

    @abc.abstractmethod
    def evaluate(self, generated_images: list[Image.Image], *args, **kwargs) -> Dict[str, float]:
        pass

    def should_trigger_for_data(self, generated_images: list[Image.Image], *args, **kwargs) -> bool:
        return True


class CLIPScoreEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.CONTROLLABILITY
    HIGHER_IS_BETTER = True

    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)

    def evaluate(self,
                 generated_images: list[Image.Image],
                 prompts: list[str],
                 **ignored_kwargs) -> Dict[str, float]:
        torch_imgs = [transforms.ToTensor()(img).to(self.device) for img in generated_images]
        self.evaluator.update(torch_imgs, prompts)
        return {"clip_score": self.evaluator.compute().item()}


class CentroidSimilarityEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.FIDELITY
    HIGHER_IS_BETTER = True

    def __init__(self, device: str):
        super().__init__(device)

    def evaluate(self,
                 generated_images: list[Image.Image],
                 real_images: list[Image.Image],
                 **ignored_kwargs) -> Dict[str, float]:
        """Returns the average cosine similarity between the generated images and the center of the cluster defined by real images."""
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        results = {}

        for encoder_cls in ALL_ENCODER_CLASSES:
            encoder = encoder_cls(self.device)
            generated_embeddings = encoder.encode(generated_images)
            generated_center = torch.mean(generated_embeddings, axis=0, keepdim=True)
            real_embeddings = encoder.encode(real_images)
            real_center = torch.mean(real_embeddings, axis=0, keepdim=True)
            results[f"centroid_similarity_{encoder.id}"] = cos(generated_center, real_center).item()
        return results


class InceptionScoreEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.IMAGE_QUALITY
    HIGHER_IS_BETTER = True

    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = InceptionScore().to(self.device)

    def evaluate(self, generated_images: list[Image.Image], **ignored_kwargs) -> Dict[str, float]:
        torch_imgs = torch.stack([transforms.ToTensor()(img).to(torch.uint8).to(self.device)
                                  for img in generated_images])
        self.evaluator.update(torch_imgs)
        mean, stddev = self.evaluator.compute()
        return {"inception_score_mean": mean.item(),
                "inception_score_stddev": stddev.item()}

    def should_trigger_for_data(self, generated_images: list[Image.Image], **ignored_kwargs) -> bool:
        # InceptionScore calculates a marginal distribution over the objects identified in the
        # images. This requires a large number of images to be useful; papers use 30k-60k images.
        # See this blog post for a high-level description of how this works:
        # https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a
        return len(generated_images) >= 1000


class FIDEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.FIDELITY
    HIGHER_IS_BETTER = False

    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator64 = FrechetInceptionDistance(feature=64).to(self.device).set_dtype(torch.float64)
        self.evaluator192 = FrechetInceptionDistance(feature=192).to(self.device).set_dtype(torch.float64)
        self.evaluator768 = FrechetInceptionDistance(feature=768).to(self.device).set_dtype(torch.float64)
        self.evaluator2048 = FrechetInceptionDistance(feature=2048).to(self.device).set_dtype(torch.float64)

    def evaluate(self,
                 generated_images: list[Image.Image],
                 real_images: list[Image.Image],
                 **ignored_kwargs) -> Dict[str, float]:
        torch_gen_imgs = torch.stack([transforms.ToTensor()(img).to(torch.uint8).to(self.device)
                                      for img in generated_images])

        # Real images (since they were not generated) might have various sizes. We'll resize them to the generated size.
        gen_size = generated_images[0].size
        real_images = [img.resize(gen_size) for img in real_images]
        torch_real_imgs = torch.stack([transforms.ToTensor()(img).to(torch.uint8).to(self.device)
                                       for img in real_images])

        self.evaluator64.update(torch_gen_imgs, real=False)
        self.evaluator64.update(torch_real_imgs, real=True)
        self.evaluator192.update(torch_gen_imgs, real=False)
        self.evaluator192.update(torch_real_imgs, real=True)
        self.evaluator768.update(torch_gen_imgs, real=False)
        self.evaluator768.update(torch_real_imgs, real=True)
        self.evaluator2048.update(torch_gen_imgs, real=False)
        self.evaluator2048.update(torch_real_imgs, real=True)
        return {"fid_score_64": self.evaluator64.compute(),
                "fid_score_192": self.evaluator192.compute(),
                "fid_score_768": self.evaluator768.compute(),
                "fid_score_2048": self.evaluator2048.compute()}

    def should_trigger_for_data(self, generated_images: list[Image.Image]):
        # Similarly to Inception Score, FID calculates marginal distributions over datasets, and is
        # only useful when there are thousands of images.
        return len(generated_images) >= 1000


class CMMDEvaluator(BaseEvaluator):
    """Original paper: https://arxiv.org/abs/2401.09603 (published Jan 2024).

    This implementation is adapted from https://github.com/sayakpaul/cmmd-pytorch/blob/main/distance.py.
    """
    TYPE = EvaluatorType.FIDELITY
    HIGHER_IS_BETTER = False
    _SIGMA = 10

    def __init__(self, device: str):
        super().__init__(device)

    def evaluate(self,
                 generated_images: list[Image.Image],
                 real_images: list[Image.Image],
                 **ignored_kwargs) -> Dict[str, float]:
        results = {}
        for encoder_cls in ALL_ENCODER_CLASSES:
            encoder = encoder_cls(self.device)
            x = encoder.encode(generated_images)
            y = encoder.encode(real_images)

            x_sqnorms = torch.diag(torch.matmul(x, x.T))
            y_sqnorms = torch.diag(torch.matmul(y, y.T))

            gamma = 1 / (2 * CMMDEvaluator._SIGMA**2)
            k_xx = torch.mean(
                torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
            )
            k_xy = torch.mean(
                torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
            )
            k_yy = torch.mean(
                torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
            )
            distance = k_xx + k_yy - 2 * k_xy
            results[f"cmmd_{encoder.id}"] = distance.item()
        return results


class AestheticPredictorEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.IMAGE_QUALITY
    HIGHER_IS_BETTER = True
    DEFAULT_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"

    def __init__(self, device: str, model_url: str = DEFAULT_URL):
        super().__init__(device)
        self.model_path = download_model(model_url, "aesthetic_predictor.pth")

    def evaluate(self, generated_images: list[Image.Image], **ignored_kwargs) -> Dict[str, float]:
        return {"aesthetic_predictor":
                run_inference(generated_images, self.model_path, self.device)}


class ImageRewardEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.CONTROLLABILITY
    HIGHER_IS_BETTER = True

    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = RM.load("ImageReward-v1.0")

    def evaluate(self,
                 generated_images: list[Image.Image],
                 prompts: list[str],
                 **ignored_kwargs) -> Dict[str, float]:
        # Returns the average image reward
        rewards = []
        for image, prompt in zip(generated_images, prompts):
            rewards.append(self.evaluator.score(prompt, image))
        return {"image_reward": sum(rewards) / len(rewards)}


class HumanPreferenceScoreEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.CONTROLLABILITY
    HIGHER_IS_BETTER = True
    DEFAULT_URL = "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?download=1"

    def __init__(self, device: str, model_url: str = DEFAULT_URL):
        super().__init__(device)
        self.hps_model_path = download_model(model_url, "human_preference_score.pt")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        if torch.cuda.is_available():
            params = torch.load(self.hps_model_path)['state_dict']
        else:
            params = torch.load(self.hps_model_path, map_location=self.device)['state_dict']
        self.model.load_state_dict(params)

    def evaluate(self,
                 generated_images: list[Image.Image],
                 prompts: list[str],
                 **ignored_kwargs) -> Dict[str, float]:
        images = [self.preprocess(img).to(self.device) for img in generated_images]
        images = torch.stack(images)
        texts = clip.tokenize(prompts, truncate=True).to(self.device)

        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        hps = image_features @ text_features.T
        hps = hps.diagonal()
        return {"human_preference_score": torch.mean(hps).detach().item()}


class VendiScoreEvaluator(BaseEvaluator):
    TYPE = EvaluatorType.DIVERSITY
    HIGHER_IS_BETTER = True

    def __init__(self, device: str):
        super().__init__(device)

    def evaluate(self, generated_images: list[Image.Image], **ignored_kwargs) -> Dict[str, float]:
        results = {}
        for encoder_cls in ALL_ENCODER_CLASSES:
            encoder = encoder_cls(self.device)
            embeddings = encoder.encode(generated_images).cpu().detach().numpy()
            results[f"vendi_score_{encoder.id}"] = vendi.score_X(embeddings).item()
        return results


def get_evaluators_for_type(evaluator_type: EvaluatorType):
    return [evaluator for evaluator in globals().values()
            if isinstance(evaluator, type)
                and hasattr(evaluator, "TYPE")
                and evaluator.TYPE == evaluator_type]
