import csv
import os
import random
from collections import defaultdict

import streamlit as st

from PIL import Image


def get_model_predictions_from_file(csv_file: str) -> dict:
    model_preds = defaultdict(list)
    # Open and read the CSV file
    with open(csv_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        _, _= header
        for row in csv_reader:
            if len(row) == 2:
                col_a_val, col_b_val = row
                model_preds["model_1"].append(col_a_val)
                model_preds["model_2"].append(col_b_val)

    return model_preds


def get_images_from_dir(dir_path: str):
    images = []
    for image in os.listdir(dir_path):
        images.append(os.path.join(dir_path, image))
    return images


model_preds = get_model_predictions_from_file("/Users/mihaileric/Documents/code/image-eval/fixture/model_comparisons.csv")
images_1 = model_preds["model_1"]
images_2 = model_preds["model_2"]

# Randomly assign one image to be A and the other to be B


image1 = Image.open(random.choice(images_1))
image2 = Image.open(random.choice(images_2))
st.session_state.image1 = image1
st.session_state.image2 = image2

# Display images
st.image(st.session_state.image1, caption="Model A")
st.image(st.session_state.image2, caption="Model B")


def update_images_displayed():
    st.session_state.image1 = random.choice(images_1)
    st.session_state.image2 = random.choice(images_2)

def compute_scores():
    pass

# Select choice for buttons
selected_option = st.radio(
    "Which image do you prefer?",
    ["**A**", "**B**"], horizontal=True)

st.button("Submit", type="primary", on_click=update_images_displayed)
st.button("Compute Score", type="primary", on_click=compute_scores)