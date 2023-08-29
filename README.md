# Image Eval

## What is this?

**TL;DR**: A Python library providing utilities to evaluate your favorite text-to-image generation models (Stable Diffusion, Midjourney, and any custom models you may have).

| Metric Name            |     Value |
|------------------------|-----------|
| clip_score             | 22.5958   |
| aesthetic_predictor    |  6.03433  |
| image_reward           | -0.661022 |
| human_preference_score |  0.196004 |
| fid                    | \[11.214, 57.471, 4.10, 442.811\] |
| inception_score        | \[1.0, 0.0\] |

![Human eval](assets/human_eval.gif)

## So, why should I care?

Since the advent of systems such as [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release), [Midjourney](https://www.midjourney.com/home/), and [DallE-2](https://openai.com/dall-e-2) text-to-image generation  
models have taken the world by storm.

However, evaluation of the quality of these systems still remains one of the
hardest challenges in continuing to improve them as there is a lack of standardization and robust tooling.

If we can't all agree on a set of practices for evaluating these systems, how can we push the state-of-the-art?

This library provides a suite of widely accepted metrics for evaluating them as well as an easy-to-use interface for performing human
evaluation of image generation model outputs.

## Cute, but what are you actually doing?

We provide six standard automated evaluation metrics out-of-the-box including:
- [CLIP Score](https://arxiv.org/abs/2104.08718)
- [Inception Score](https://arxiv.org/abs/1606.03498)
- [FID](https://arxiv.org/abs/1512.00567)
- [Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [Image Reward](https://arxiv.org/pdf/2304.05977.pdf)
- [Human Preference Score](https://tgxs002.github.io/align_sd_web/)

We also provide a simple and ready-to-use [Streamlit](https://streamlit.io/) interface for performing human evaluation of model outputs on your local machine.


## Installation

This library has been tested on Python 3.9.12. Installing the library involves running:
```
pip install image-eval
pip install git+https://github.com/openai/CLIP.git # A dependency some metrics require
```

Optionally, if you have a CUDA-enabled device, install the [version of PyTorch](https://pytorch.org/get-started/previous-versions/) that matches your CUDA version. For CUDA 11.3, that might look like:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

NOTE: If you want to use the `aesthetic_predictor` and `human_preference_score` metrics, you will need to download the respective model weights.

We require having a `models` folder and then having a separate subfolder for `aesthetic_predictor` and `human_preference_score` with the downloaded weights.

For `aesthetic_predictor` you can download the weights of the model [here](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth).

For `human_preference_score` you can download the weights of the model [here](https://mycuhk-my.sharepoint.com/personal/1155172150_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155172150%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FHPS%2Fhpc%2Ept&parent=%2Fpersonal%2F1155172150%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FHPS&ga=1).

Your `models` folder should then have this structure:
```
models/
├── aesthetic_predictor/
│   └── sac+logos+ava1-l14-linearMSE.pth
└── human_preference_score/
    └── hpc.pt
```

You can then specify your `models` folder in the [usage](#usage) section below.

## Usage

There are two ways to interact with the `image-eval` library: either through the CLI or through the API.

### CLI

You can invoke all the metric computations through the CLI once you've `pip install`'d the library. The library makes certain
assumptions about the format of the inputs to the CLI.

For example, if you want to calculate a metric assessing the match between
prompts and generated images (as is the case with `clip_score`), you would invoke:
```
image_eval -m clip_score -p /path/to/image_to_prompt.json -g /path/to/folder/with/generated/images 
```

Here `image_to_prompt.json` is a JSON file with the following format:
```
{
    "image_1.jpg": "prompt for image 1",
    "image_2.jpg": "prompt for image 2",
    ...
}
```

where `image_1.jpg` and `image_2.jpg` are the names of the generated images in the folder specified by the `-g` flag.

If you want to calculate a metric assessing the match between generated images and a set of reference images (as is the case with `fid`), you would invoke:
```
image_eval -m fid -g /path/to/generated/images -r /path/to/real/images
```

Some metrics may need you to specify additional arguments. For example, if you want to use the `aesthetic_predictor` or `human_preference_score` metrics, you would invoke:
```
image_eval -m aesthetic_predictor -p /path/to/image_to_prompt.json -g /path/to/generated/images --model-dir /path/to/folder/with/models
```

You can also compute multiple metrics simultaneously by passing in a comma-separated list of metrics to the `-m` flag:
```
image_eval -m clip_score,fid -p /path/to/image_to_prompt.json -g /path/to/generated/images -r /path/to/real/images
```

If you want to launch the human evaluation interface, you would invoke:
```
image_eval --local-human-eval --model-predictions-json /path/to/model_comparisons.json
```

Here `model_comparisons.json` is a JSON file with the following format:
```
[
    {
        "model_1": "path to image 1 from model 1",
        "model_2": "path to image 1 from model 2",
        "prompt": "prompt for image 1"
    },
    {
        "model_1": "path to image 2 from model 1",
        "model_2": "path to image 2 from model 2",
        "prompt": "prompt for image 2"
    },
    ...
]
```

where `model_1` and `model_2` are the keys for the paths to image outputs for the respective models. Our library does expect the **keys to match these values exactly**.

An interface should launch in your browser at `http://localhost:8501`.

NOTE: When you click `Compute Model Wins` a local file named `scores.json` will be created in the directory from which you launched the CLI.

### Programmatic

You can also interact with the library through the API directly. For example if you want to invoke the `clip_score` metric, you would do the following:
```
from image_eval.evaluators import CLIPScoreEvaluator

evaluator = CLIPScoreEvaluator(device="cpu") # or "cuda" if you have a GPU-enabled device
images = [np.randint(0, 255, (224, 224, 3)) for _ in range(10)] # list of 10 random images
prompts = ["random prompt" * 10]
evaluator.evaluate(images, prompts)
```

It's that simple!

## Contributing

We welcome any and all contributions to this library! We have a wishlist of items we'd like to add to the library [here](#roadmap).

### How to add your own automated metric

We designed the library to make it easy to add your own automated metric. All you need to do is create a new class that inherits from the `BaseReferenceFreeEvaluator` or `BaseWithReferenceEvaluator` class and implements the `evaluate` method.

For examples of how our current metrics implement this contract, see `evaluators.py`.

### Roadmap
- [ ] Make it easy to launch remote human evaluation task using [AWS Groundtruth](https://aws.amazon.com/sagemaker/data-labeling/)
- [ ] Add ability to log metrics to [WandB](https://wandb.ai/site)
