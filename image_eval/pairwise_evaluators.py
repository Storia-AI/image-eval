import torch

from PIL import Image
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import UniversalImageQualityIndex
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from typing import Any, Dict, List

from image_eval.evaluators import BaseEvaluator
from image_eval.evaluators import EvaluatorType


class PairwiseSimilarityEvaluator(BaseEvaluator):
    """Evaluates how similar pairs of images are.

    Helpful to compare against images generated from consecutive checkpoints during training. When
    these metrics start signaling high similarity between consecutive checkpoints, it's a sign we
    can stop training, since the model has converged.

    All the children of these methods are metrics found in HEIM:
    https://crfm.stanford.edu/helm/heim/latest/
    """
    TYPE = EvaluatorType.PAIRWISE_SIMILARITY

    def __init__(self, device: str, metric_name: str, metric: Any, normalize: bool = False):
        """Constructs a pairwise evaluator that uses a given metric from `torchmetrics`.

        Args:
            device: "cpu" or "cuda"
            metric_name: A human-readable name for the metric
            metric: An object that has a __call__(images1, images2) method for pairwise comparison
            normalize: Whether images need to be normalized to [0, 1] before passing to the metric
        """
        super().__init__(device)
        self.metric_name = metric_name
        self.metric = metric.to(device)
        self.normalize = normalize

    def evaluate(self,
                 generated_images: List[Image.Image],
                 real_images: List[Image.Image],
                 **unused_kwargs) -> Dict[str, float]:
        if len(generated_images) != len(real_images):
            raise ValueError("Pairwise evaluators expect 1:1 pairs of generated/real images.")

        img_transforms = [transforms.Resize((256, 256)), transforms.ToTensor()]
        if self.normalize:
            img_transforms.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        preprocessing = transforms.Compose(img_transforms)
        generated_images = [preprocessing(img) for img in generated_images]
        real_images = [preprocessing(img) for img in real_images]

        generated_images = torch.stack(generated_images).to(self.device)
        real_images = torch.stack(real_images).to(self.device)

        score = self.metric(generated_images, real_images).detach().item()
        return {self.metric_name: score}


class LPIPSEvaluator(PairwiseSimilarityEvaluator):
    """Calculates Learned Perceptual Image Patch Similarity (LPIPS) score.

    It computes the distance between the activations of two image patches for some deep network
    (by default, we use VGG). The score is between 0 and 1, where 0 means the images are identical.

    Original paper: https://arxiv.org/pdf/1801.03924 (published Apr 2018).
    """
    HIGHER_IS_BETTER = False

    def __init__(self, device):
        super().__init__(device, "LPIPS", LearnedPerceptualImagePatchSimilarity(net_type="vgg"),
                         normalize=True)


class MultiSSIMEvaluator(PairwiseSimilarityEvaluator):
    """Calculates Multi-scale Structural Similarity Index Measure (SSIM).

    This is an extension of SSIM, which assesses the similarity between two images based on three
    components: luminance, contrast, and structure. The score is between -1 and 1, where 1 = perfect
    similarity, 0 = no similarity, -1 = perfect anti-corelation.

    Original paper: https://ieeexplore.ieee.org/document/1292216 (published Apr 2004).
    """
    HIGHER_IS_BETTER = True

    def __init__(self, device):
        super().__init__(device, "MultiSSIM", MultiScaleStructuralSimilarityIndexMeasure())


class PSNREvaluator(PairwiseSimilarityEvaluator):
    """Calculates Peak Signal-to-Noise Ratio (PSNR).

    It was originally designed to measure the quality of reconstructed or compressed images compared
    to their original versions. Its values are between -infinity and +infinity, where identical
    images score +infinity.

    Original paper: https://ieeexplore.ieee.org/document/1163711 (published Sep 2000).
    """
    HIGHER_IS_BETTER = True

    def __init__(self, device):
        super().__init__(device, "PSNR", PeakSignalNoiseRatio())


class UIQIEvaluator(PairwiseSimilarityEvaluator):
    """Calculates Universal Image Quality Index (UIQI).

    Based on the idea of comparing statistical properties of an original and a distorted image in
    both the spatial and frequency domains. The calculation involves several steps, including the
    computation of mean, variance, and covariance of the pixel values in local windows of the
    images. It also considers factors like luminance, contrast, and structure.

    Original paper: https://ieeexplore.ieee.org/document/1284395 (published Jul 2004).
    """
    def __init__(self, device):
        super().__init__(device, "UIQUI", UniversalImageQualityIndex())
