import abc
from typing import Optional

import ImageReward as RM
import clip
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from transformers import AutoImageProcessor
from transformers import Dinov2Model
from transformers import CLIPModel
from transformers import CLIPProcessor

from image_eval.improved_aesthetic_predictor import run_inference

torch.manual_seed(42)


# TODO (mihail): Decouple this so not all evaluators are in the same file

class BaseReferenceFreeEvaluator(abc.ABC):
    """
    An evaluation that doesn't require gold samples to compare against.
    """
    def __init__(self, device: str, model_path: Optional[str] = None):
        self.device = device
        self.model_path = model_path

    @abc.abstractmethod
    def evaluate(self, images: list[Image.Image], prompts: list[str]):
        pass


class BaseWithReferenceEvaluator(abc.ABC):
    """
    An evaluation that includes gold samples to compare against.
    """
    def __init__(self, device: str, model_path: Optional[str] = None):
        self.device = device
        self.model_path = model_path

    @abc.abstractmethod
    def evaluate(self, generated_images: list[Image.Image], real_images: list[Image.Image]):
        pass


class CLIPScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)

    def evaluate(self, images: list[Image.Image], prompts: list[str]):
        torch_imgs = [transforms.ToTensor()(img).to(self.device) for img in images]
        self.evaluator.update(torch_imgs, prompts)
        return self.evaluator.compute()


class CLIPSimilarityEvaluator(BaseWithReferenceEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        model_name = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def evaluate(self, generated_images: list[Image.Image], real_images: list[Image.Image]):
        """Returns the average similarity between the generated images and the center of the cluster defined by real images."""
        generated_images_inputs = self.processor(text=None, images=generated_images, return_tensors="pt", padding=True)
        generated_images_inputs = {k: v.to(self.device) for k, v in generated_images_inputs.items()}
        generated_images_embeddings = self.model.get_image_features(**generated_images_inputs)
        generated_images_center = torch.mean(generated_images_embeddings, axis=0, keepdim=True)

        real_images_inputs = self.processor(text=None, images=real_images, return_tensors="pt", padding=True)
        real_images_inputs = {k: v.to(self.device) for k, v in real_images_inputs.items()}
        real_images_embeddings = self.model.get_image_features(**real_images_inputs)
        real_images_center = torch.mean(real_images_embeddings, axis=0, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(real_images_center, generated_images_center)


class DinoV2SimilarityEvaluator(BaseWithReferenceEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        model_name = "facebook/dinov2-base"
        self.model = Dinov2Model.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def evaluate(self, generated_images: list[Image.Image], real_images: list[Image.Image]):
        """Returns the average similarity between the generated images and the center of the cluster defined by real images."""
        generated_images_inputs = self.processor(text=None, images=generated_images, return_tensors="pt", padding=True)
        generated_images_inputs = {k: v.to(self.device) for k, v in generated_images_inputs.items()}
        generated_images_embeddings = self.model(**generated_images_inputs).pooler_output
        generated_images_center = torch.mean(generated_images_embeddings, axis=0, keepdim=True)

        real_images_inputs = self.processor(text=None, images=real_images, return_tensors="pt", padding=True)
        real_images_inputs = {k: v.to(self.device) for k, v in real_images_inputs.items()}
        real_images_embeddings = self.model(**real_images_inputs).pooler_output
        real_images_center = torch.mean(real_images_embeddings, axis=0, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(real_images_center, generated_images_center)


class InceptionScoreEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator = InceptionScore().to(self.device)

    def evaluate(self, images: list[Image.Image], ignored_prompts: list[str]):
        torch_imgs = torch.stack([transforms.ToTensor()(img).to(torch.uint8).to(self.device) for img in images])
        self.evaluator.update(torch_imgs)
        return self.evaluator.compute()


class FIDEvaluator(BaseWithReferenceEvaluator):
    def __init__(self, device: str):
        super().__init__(device)
        self.evaluator64 = FrechetInceptionDistance(feature=64).to(self.device).set_dtype(torch.float64)
        self.evaluator192 = FrechetInceptionDistance(feature=192).to(self.device).set_dtype(torch.float64)
        self.evaluator768 = FrechetInceptionDistance(feature=768).to(self.device).set_dtype(torch.float64)
        self.evaluator2048 = FrechetInceptionDistance(feature=2048).to(self.device).set_dtype(torch.float64)

    def evaluate(self, generated_images: list[Image.Image], real_images: list[Image.Image]):
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
        return (self.evaluator64.compute(),
                self.evaluator192.compute(),
                self.evaluator768.compute(),
                self.evaluator2048.compute())


class AestheticPredictorEvaluator(BaseReferenceFreeEvaluator):
    def __init__(self, device: str, model_path: str):
        super().__init__(device, model_path)

    def evaluate(self, images: list[Image.Image], ignored_prompts: list[str]):
        return run_inference(images, self.model_path, self.device)


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
    def __init__(self, device: str, model_path: str):
        super().__init__(device, model_path)
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def evaluate(self, images: list[Image.Image], prompts: list[str]):
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
