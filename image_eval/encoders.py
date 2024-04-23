import abc

from PIL import Image
from transformers import AutoImageProcessor
from transformers import Dinov2Model
from transformers import CLIPModel
from transformers import CLIPProcessor
from transformers import ConvNextV2Model


class BaseEncoder(abc.ABC):
    """A model that maps images from pixel space to an embedding."""
    def __init__(self, id: str, device: str):
        self.id = id
        self.device = device

    @abc.abstractmethod
    def encode(self, images: list[Image.Image]):
        pass


class CLIPEncoder(BaseEncoder):
    """
    Original paper: https://arxiv.org/abs/2103.00020 (published Feb 2021).
    Used by LyCORIS for evaluation: https://arxiv.org/pdf/2309.14859.pdf.
    """
    def __init__(self, device: str):
        super().__init__("clip", device)
        model_name = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, images: list[Image.Image]):
        image_inputs = self.processor(text=None, images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        return self.model.get_image_features(**image_inputs)


class DinoV2Encoder(BaseEncoder):
    """
    Original paper: https://arxiv.org/abs/2304.07193 (published April 2023).
    Used by LyCORIS for evaluation: https://arxiv.org/pdf/2309.14859.pdf.
    """
    def __init__(self, device: str):
        super().__init__("dino_v2", device)
        model_name = "facebook/dinov2-base"
        self.model = Dinov2Model.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def encode(self, images: list[Image.Image]):
        image_inputs = self.processor(text=None, images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        return self.model(**image_inputs).pooler_output


class ConvNeXtV2Encoder(BaseEncoder):
    """
    Original paper: https://arxiv.org/abs/2301.00808 (published Jan 2023).
    Used by LyCORIS for evaluation: https://arxiv.org/pdf/2309.14859.pdf.
    """
    def __init__(self, device: str):
        super().__init__("convnext_v2", device)
        model_name = "facebook/convnextv2-base-22k-384"
        self.model = ConvNextV2Model.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def encode(self, images: list[Image.Image]):
        image_inputs = self.processor(text=None, images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        return self.model(**image_inputs).pooler_output


ALL_ENCODER_CLASSES = [CLIPEncoder, DinoV2Encoder, ConvNeXtV2Encoder]
