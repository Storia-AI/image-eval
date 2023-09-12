
## What is this?

## So, why should I care?



## Installation


## Usage

### CLI


### Programmatic

## Contributing




## Roadmap
- [ ] Make it easy to launch remote human evaluation task using [AWS Groundtruth](https://aws.amazon.com/sagemaker/data-labeling/)
- [ ] Add ability to log metrics to [wandb](https://wandb.ai/site)


# Notes
- [https://huyenchip.com/2023/08/16/llm-research-open-challenges.html](https://huyenchip.com/2023/08/16/llm-research-open-challenges.html)
   - https://blog.yenniejun.com/p/all-languages-are-not-created-tokenized
- [https://eugeneyan.com/writing/llm-patterns/](https://eugeneyan.com/writing/llm-patterns/)
  - https://tatsu-lab.github.io/alpaca_eval/
    - LLM-based automatic eval that is cheap, fast, and reliable
    - Thus, instead of using off-the-shelf benchmarks, we can start by collecting a set of task-specific evals (i.e., prompt, context, expected outputs as references). These evals will then guide prompt engineering, model selection, fine-tuning, and so on. And as we update our systems, we can run these evals to quickly measure improvements or regressions. Think of it as Eval Driven Development (EDD).
    - Rather than asking an LLM for a direct evaluation (via giving a score), try giving it a reference and asking for a comparison
- [https://arxiv.org/pdf/2307.10169.pdf]
  - Challenges in LLMs
    - unfathomable datasets
    - unsustainable loss power-law
    - when adapting an LLM via full model fine-tuning, individual copy of model must be stored/loaded

- Automated image generation metrics
  - Inception score
    - Obtain probability distribution by running through InceptionV3 and measure peakiness of distribution
  - FID
    - compares mean and standard deviation of gaussian distributions for feature vectors obtained from deepest layer in Inception v3
    - low FID is good
  - Structured similarity index metric
    - computes luminance, contrast, and structure
  - https://huggingface.co/docs/diffusers/conceptual/evaluation
    - qualitative evaluation
      - compositionality, image-text alignment, spatial relations
      - DrawBench and PartiPrompts as datasets for qualitative benchmarking
    - quantitative evaluation
      - CLIP score / RefCLIPScore (when augmented with references)
        - reference in article
        - TODO: Include option for multiple images per prompt
      - for text + image -> image
        - CLIP directional similarity
          - measure consistency of change between two images and two image captions
      - torchmetrics includes FID + Inception Distance + CLIP scores
  - https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models
    - CLIPScore
    - Improved Aesthetic Predictor
    - ImageReward
    - Human Preference Score
    - X-IQE
  - https://arxiv.org/pdf/2307.04749.pdf
    - break prompt down into disjoint assertion sets
  - https://github.com/layer6ai-labs/dgm-eval
    - library of evaluation metrics

# TODO
- Add a download models step first in the README
