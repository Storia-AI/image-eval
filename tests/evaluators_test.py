"""
To run this test:

pip install pytest
export PYTHONPATH=$PYTHONPATH:/path/to/parent/of/image-eval
pytest evaluators_test.py --log-cli-level=INFO
"""
import logging
import torch
import unittest

import sys
sys.path.append("../image_eval")

from PIL import Image
from image_eval.evaluators import (
    AestheticPredictorEvaluator,
    EvaluatorType,
    get_evaluators_for_type,
)

class TestEvaluators(unittest.TestCase):
    """Unit tests the evaluators by comparing scores of different image sets."""

    @staticmethod
    def _single_eval(evaluator, image, other_image=None, prompt=None):
        """Extracts the score from the result of an evaluator, which is a {metric:score} dict."""
        result = evaluator.evaluate([image], real_images=[other_image], prompts=[prompt])
        assert len(result) == 1
        return list(result.values())[0]

    def test_image_quality_evaluators(self):
        """Tests that an original image scores higher than one with added Gaussian noise."""
        img_orig = Image.open("assets/julie.jpeg")
        img_noise1 = Image.open("assets/julie_gn_mean0_sigma50.jpeg")
        img_noise2 = Image.open("assets/julie_gn_mean0_sigma100.jpeg")

        # TODO(julia): Add InceptionScoreEvaluator.
        for evaluator_cls in [AestheticPredictorEvaluator]:
            evaluator = evaluator_cls("cuda" if torch.cuda.is_available() else "cpu")
            score_orig = self._single_eval(evaluator, img_orig)
            score_noise1 = self._single_eval(evaluator, img_noise1)
            score_noise2 = self._single_eval(evaluator, img_noise2)
            logging.info(f"Scores from {evaluator}: {score_orig}, {score_noise1}, {score_noise2}")

            if evaluator_cls.HIGHER_IS_BETTER:
                self.assertGreater(score_orig, score_noise1)
                self.assertGreater(score_noise1, score_noise2)
            else:
                self.assertLess(score_orig, score_noise1)
                self.assertLess(score_noise1, score_noise2)

    def test_controllability_evaluators(self):
        """Tests that an image scores higher on a descriptive prompt than on a random prompt."""
        img_dog = Image.open("assets/julie.jpeg")

        for evaluator_cls in get_evaluators_for_type(EvaluatorType.CONTROLLABILITY):
            evaluator = evaluator_cls("cuda" if torch.cuda.is_available() else "cpu")
            score_good_prompt = self._single_eval(evaluator, img_dog, prompt="a dog")
            score_bad_prompt = self._single_eval(evaluator, img_dog, prompt="a house")
            logging.info(f"Scores from {evaluator}: "
                         f"good_prompt={score_good_prompt}, bad_prompt={score_bad_prompt}")

            if evaluator_cls.HIGHER_IS_BETTER:
                self.assertGreater(score_good_prompt, score_bad_prompt)
            else:
                self.assertLess(score_good_prompt, score_bad_prompt)

    def test_fidelity_evaluators(self):
        """Tests that datasets with similar images score higher than ones with different images."""
        img_julie1 = Image.open("assets/julie.jpeg")
        img_julie2 = Image.open("assets/julie_gn_mean0_sigma50.jpeg")
        img_other = Image.open("assets/fortune_teller.png")

        for evaluator_cls in get_evaluators_for_type(EvaluatorType.FIDELITY):
            evaluator = evaluator_cls("cuda" if torch.cuda.is_available() else "cpu")

            # We're adding 2 images to each dataset, because some evaluators require at least 2.
            results_similar = evaluator.evaluate(generated_images=[img_julie2] * 2,
                                                 real_images=[img_julie1] * 2)
            results_dissimilar = evaluator.evaluate(generated_images=[img_julie2] * 2,
                                                    real_images=[img_other] * 2)

            # The same evaluator might return multiple metrics.
            for similar_key, similar_value in results_similar.items():
                # TODO(julia & vlad): Figure out why insightface often fails or returns an empty
                # result even on pictures with human faces.
                if "insightface" in similar_key:
                    continue

                dissimilar_value = results_dissimilar[similar_key]
                logging.info(f"Scores from {evaluator}/{similar_key}: "
                             f"similar={similar_value}, dissimilar={dissimilar_value}")

                if evaluator_cls.HIGHER_IS_BETTER:
                    self.assertGreater(similar_value, dissimilar_value)
                else:
                    self.assertLess(similar_value, dissimilar_value)

    def test_diversity_evaluators(self):
        """Tests that a dataset with identical images scores higher than one with diverse images."""
        img_julie1 = Image.open("assets/julie.jpeg")
        img_julie2 = Image.open("assets/julie_gn_mean0_sigma50.jpeg")
        img_other = Image.open("assets/fortune_teller.png")

        for evaluator_cls in get_evaluators_for_type(EvaluatorType.DIVERSITY):
            evaluator = evaluator_cls("cuda" if torch.cuda.is_available() else "cpu")

            results_not_diverse = evaluator.evaluate([img_julie1, img_julie2])
            results_diverse = evaluator.evaluate([img_julie1, img_other])

            for key_not_diverse, value_not_diverse in results_not_diverse.items():
                # TODO(julia & vlad): Figure out why insightface often fails or returns an empty
                # result even on pictures with human faces.
                if "insightface" in key_not_diverse:
                    continue

                value_diverse = results_diverse[key_not_diverse]
                logging.info(f"Scores from {evaluator}/{key_not_diverse}: "
                             f"not_diverse={value_not_diverse}, diverse={value_diverse}")

                if evaluator_cls.HIGHER_IS_BETTER:
                    self.assertGreater(value_diverse, value_not_diverse)
                else:
                    self.assertLess(value_diverse, value_not_diverse)


if __name__ == "__main__":
    unittest.main()
