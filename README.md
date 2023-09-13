# Image Eval

## What is this?

**TL;DR**: A Python library providing utilities to evaluate your favorite text-to-image generation models (Stable Diffusion, Midjourney, and any custom models you may have). This includes six standard automated evaluation metrics and a system for doing human evaluation of images.

| Metric Name            |     Value |
|------------------------|-----------|
| clip_score             | 22.5958   |
| aesthetic_predictor    |  6.03433  |
| image_reward           | -0.661022 |
| human_preference_score |  0.196004 |
| fid                    | [11.214, 57.471, 4.10, 442.811] |
| inception_score        | [1.0, 0.0] |

![](assets/human_eval.gif)

## So, why should I care?



## Installation


## Usage

### CLI


### Programmatic

## Contributing




## Roadmap
- [ ] Make it easy to launch remote human evaluation task using [AWS Groundtruth](https://aws.amazon.com/sagemaker/data-labeling/)
- [ ] Add ability to log metrics to [WandB](https://wandb.ai/site)




# TODO
- Add a download models step first in the README
