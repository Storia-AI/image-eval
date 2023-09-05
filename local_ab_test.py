import csv
import json
import os
import random
from collections import defaultdict

import streamlit as st
from PIL import Image


def get_model_predictions_from_file(json_file: str) -> dict:
    model_preds = defaultdict(list)
    with open(json_file) as f:
        data = json.load(f)
        for entry in data:
            model_preds["model_1"].append(entry["model_1"])
            model_preds["model_2"].append(entry["model_2"])

    return model_preds


def get_images_from_dir(dir_path: str):
    images = []
    for image in os.listdir(dir_path):
        images.append(os.path.join(dir_path, image))
    return images


model_preds = get_model_predictions_from_file("/home/venus/Documents/code/image-eval/fixture/model_comparisons.json")
images_1 = model_preds["model_1"]
images_2 = model_preds["model_2"]


def assign_images():
    # Randomly assign one image to be A
    image_1 = images_1[st.session_state.curr_idx]
    image_2 = images_2[st.session_state.curr_idx]

    print(image_1, image_2)

    image_a = random.choice([image_1, image_2])
    if image_a == image_1:
        image_b = image_2
    else:
        image_b = image_1

    image_a = Image.open(image_a)
    image_b = Image.open(image_b)
    st.session_state.image_a = image_a
    st.session_state.image_b = image_b


# Initialize session state
if "curr_idx" not in st.session_state:
    st.session_state.curr_idx = 0
if "image_a" not in st.session_state:
    assign_images()


def update_images_displayed():
    st.session_state.curr_idx += 1
    assign_images()
    print(f"CURR IDX: {st.session_state.curr_idx}")


def compute_scores():
    pass


col1, col2 = st.columns(2)
# Display images
with col1:
    st.image(st.session_state.image_a, caption="Model A")
with col2:
    st.image(st.session_state.image_b, caption="Model B")
st.write(f"Processing {st.session_state.curr_idx + 1} / {len(images_1)} images")
# Select choice for buttons
selected_option = st.radio(
    "Which image do you prefer?",
    ["**A**", "**B**"], horizontal=True)
print(selected_option)

st.button("Submit", type="primary", on_click=update_images_displayed)
st.button("Compute Score", type="primary", on_click=compute_scores)
