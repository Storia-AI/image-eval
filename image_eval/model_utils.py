"""Utilities for manipulating models."""
import logging
import os
import requests

from pathlib import Path


def download_model(url: str, output_filename: str, output_dir: str = "~/.cache"):
    """Downloads a file from the given URL. Returns the output path."""
    if output_dir == "~/.cache":
        home = str(Path.home())
        output_dir = os.path.join(home, ".cache")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        logging.info(f"Reusing model cached at {output_path}.")
        return output_path

    logging.info(f"Downloading model from {url}...")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url, stream=True)
    if not response.status_code == 200:
        raise RuntimeError(f"Unable to download model from {url}.")

    with open(output_path, "wb") as f:
        # Iterate over the response content in chunks
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return output_path
