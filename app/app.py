from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_model, predict
from utils import MODEL_PATH, prepare_image


@st.cache_resource(show_spinner=False)
def warm_model() -> bool:
    load_model(MODEL_PATH)
    return True


def main() -> None:
    st.set_page_config(page_title="DigitSense AI", layout="centered")
    st.title("DigitSense AI")
    st.write("Upload a handwritten digit image and run a prediction from the trained MNIST model.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Train the model with `python src/train.py`, then upload an image to predict.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=224)

    if st.button("Run prediction", type="primary"):
        try:
            warm_model()
            predicted_digit, probabilities = predict(prepare_image(image), MODEL_PATH)
        except FileNotFoundError:
            st.warning("No trained model was found at `outputs/model.pth`. Run `python src/train.py` first.")
            return

        confidence = float(probabilities[predicted_digit].item()) * 100.0
        st.markdown(f"## Predicted Digit: {predicted_digit}")
        st.metric("Confidence", f"{confidence:.2f}%")

        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.bar(range(10), probabilities.numpy(), color="#0F766E")
        axis.set_xticks(range(10))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Digit")
        axis.set_ylabel("Probability")
        axis.set_title("Class Probabilities")
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)


if __name__ == "__main__":
    main()
