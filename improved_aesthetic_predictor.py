import PIL
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

import clip

from PIL import Image

"""
Adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/tree/main
"""

#####  This script will predict the aesthetic score for this image file:

# If you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def run_inference(images: list[Image]):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    # load the model you trained previously or the model available in this repo
    if torch.cuda.is_available():
        s = torch.load(
            "models/sac+logos+ava1-l14-linearMSE.pth")
    else:
        s = torch.load(
            "models/sac+logos+ava1-l14-linearMSE.pth", map_location=torch.device("cpu"))

    model.load_state_dict(s)

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
    avg_aesthetic_score = 0
    for pil_image in images:
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        if torch.cuda.is_available():
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        else:
            prediction = model(torch.from_numpy(im_emb_arr).to(device))

        avg_aesthetic_score += prediction.item()

    return avg_aesthetic_score / len(images)

images = ["/Users/mihaileric/Documents/code/image-eval/fixture/images/480749_dog in slacks _xl-1024-v1-0.png"]
imgs = []
for img_path in images:
    imgs.append(Image.open(img_path).convert("RGB"))

run_inference(imgs)