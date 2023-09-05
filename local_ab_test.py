import argparse
import json
import random
from collections import Counter
from collections import defaultdict

import streamlit as st
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model-predictions", "-m", help="path to json file containing model predictions")
args = parser.parse_args()


def get_model_predictions_from_file(json_file: str) -> dict:
    model_preds = defaultdict(list)
    with open(json_file) as f:
        data = json.load(f)
        for entry in data:
            model_preds["model_1"].append(entry["model_1"])
            model_preds["model_2"].append(entry["model_2"])

    assert len(model_preds["model_1"]) == len(model_preds["model_2"]), \
        "You must ensure you have an equal number of predictions per model"
    return model_preds


model_preds = get_model_predictions_from_file(args.model_predictions)
images_1 = model_preds["model_1"]
images_2 = model_preds["model_2"]


def assign_images():
    # Randomly assign one image to be A
    image_1 = images_1[st.session_state.curr_idx]
    image_2 = images_2[st.session_state.curr_idx]

    image_a = random.choice([image_1, image_2])
    if image_a == image_1:
        st.session_state.model_a_assignments.append("model_1")
        image_b = image_2
    else:
        st.session_state.model_a_assignments.append("model_2")
        image_b = image_1

    image_a = Image.open(image_a)
    image_b = Image.open(image_b)
    st.session_state.image_a = image_a
    st.session_state.image_b = image_b


# Initialize session state
if "curr_idx" not in st.session_state:
    st.session_state.curr_idx = 0
    st.session_state.click_disabled = False
    st.session_state.model_a_assignments = []
    st.session_state.model_wins = Counter()
    st.session_state.wins_str = ""
if "image_a" not in st.session_state:
    assign_images()


def update_images_displayed():
    if not st.session_state.click_disabled:
        # Increment model wins
        if st.session_state.curr_idx < len(images_1):
            if selected_option == "**A**":
                st.session_state.model_wins[st.session_state.model_a_assignments[st.session_state.curr_idx]] += 1
            else:
                if st.session_state.model_a_assignments[st.session_state.curr_idx] == "model_1":
                    model_to_increment = "model_2"
                else:
                    model_to_increment = "model_1"
                st.session_state.model_wins[model_to_increment] += 1
        st.session_state.curr_idx += 1
    if st.session_state.curr_idx >= len(images_1):
        st.session_state.click_disabled = True
    else:
        assign_images()


def compute_scores_and_dump():
    model_1_wins = st.session_state.model_wins["model_1"] / len(images_1)
    model_2_wins = st.session_state.model_wins["model_2"] / len(images_1)
    with open("scores.json", "w") as f:
        json.dump(st.session_state.model_wins, f)
    st.session_state.wins_str = f"Model 1 wins %: {model_1_wins * 100}, " \
                                f"Model 2 wins %: {model_2_wins * 100}"


col1, col2 = st.columns(2)
# Display images
with col1:
    st.image(st.session_state.image_a, caption="Model A")
with col2:
    st.image(st.session_state.image_b, caption="Model B")
st.write(
    f"Processing {st.session_state.curr_idx + 1 if not st.session_state.click_disabled else st.session_state.curr_idx} "
    f"/ {len(images_1)} images")
# Select choice for buttons
selected_option = st.radio("Which image do you prefer?", ["**A**", "**B**"], horizontal=True)

st.button("Submit", type="primary", on_click=update_images_displayed, disabled=st.session_state.click_disabled)
st.button("Compute Model Wins", type="primary", on_click=compute_scores_and_dump)
st.write(f"{st.session_state.wins_str}")
