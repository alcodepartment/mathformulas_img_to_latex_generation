import io
import os
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

import torch
from torch import nn

from model import Im2Latex
from tokenizer import Tokenizer

DEFAULT_IMG_HEIGHT = 64
DEFAULT_MAX_LEN = 150
CKPT_PATH = "./checkpoints/exp1/best4.pt"
VOCAB_PATH: Optional[str] = './vocab.json'

@st.cache_resource
def load_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource
def load_tokenizer() -> Tokenizer:
    if VOCAB_PATH is None:
        st.error(f"Vocab file not found at {VOCAB_PATH}")
        raise FileNotFoundError(VOCAB_PATH)

    tokenizer = Tokenizer.load(str(VOCAB_PATH))
    return tokenizer



@st.cache_resource
def load_model(device: torch.device, vocab_size: int) -> Im2Latex:
    """
    Create the model and load checkpoint.
    """
    model = Im2Latex(
        vocab_size=vocab_size,
        encoder_type="exp",
        attn_size=128,
    ).to(device)

    if os.path.isfile(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        st.warning(f"Checkpoint not found at {CKPT_PATH}. Model is randomly initialized.")

    model.eval()
    return model



def preprocess_image(
    pil_img: Image.Image,
    img_height: int = DEFAULT_IMG_HEIGHT
) -> torch.Tensor:
    """
    Preprocess a PIL Image to (1, 1, H, W) tensor for the model.

    - grayscale
    - resize to fixed height, keep aspect ratio
    - normalize to [0,1]
    """
    pil_img = pil_img.convert("L")
    w, h = pil_img.size

    new_h = img_height
    new_w = int(w * (new_h / h)) if h > 0 else img_height

    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(pil_img).astype("float32") / 255.0  # (H,W)

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor


def decode_prediction(
    model: Im2Latex,
    tokenizer: Tokenizer,
    image_tensor: torch.Tensor,
    device: torch.device,
    max_len: int = DEFAULT_MAX_LEN,
) -> str:
    """
    Run model.greedy_decode on a single image and decode to LaTeX string.
    """
    image_tensor = image_tensor.to(device)

    pad_id = tokenizer.token2id[tokenizer.pad_token]
    sos_id = tokenizer.token2id[tokenizer.start_token]
    eos_id = tokenizer.token2id[tokenizer.end_token]

    with torch.no_grad():
        pred_ids = model.greedy_decode(
            image_tensor,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            mem_pad=None,
        )  # (1, L_pred)

    seq = pred_ids[0].tolist()
    # strip PAD, cut at EOS
    seq = [tok for tok in seq if tok != pad_id]
    if eos_id in seq:
        seq = seq[: seq.index(eos_id) + 1]

    latex_str = tokenizer.decode(seq, remove_special=True)
    return latex_str

def main():
    st.set_page_config(page_title="Im2LaTeX Demo", layout="wide")

    st.title("🧮 Image → LaTeX OCR (Im2LaTeX demo)")

    st.markdown(
        """
        Upload an image of a mathematical formula and the model will attempt
        to generate the corresponding **LaTeX code**.
        """
    )

    device = load_device()
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.token2id)
    model = load_model(device, vocab_size)


    st.sidebar.header("Settings")
    img_height = st.sidebar.number_input(
        "Image height (used in preprocessing)",
        min_value=32,
        max_value=256,
        value=DEFAULT_IMG_HEIGHT,
        step=8,
    )
    max_len = st.sidebar.number_input(
        "Max decoding length",
        min_value=30,
        max_value=400,
        value=DEFAULT_MAX_LEN,
        step=10,
    )

    uploaded_file = st.file_uploader(
        "Upload formula image (png/jpg)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Load image
        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input image")
            st.image(pil_img, use_column_width=True)

        # Preprocess
        image_tensor = preprocess_image(pil_img, img_height=img_height)

        # Predict
        with st.spinner("Running model..."):
            latex_pred = decode_prediction(
                model=model,
                tokenizer=tokenizer,
                image_tensor=image_tensor,
                device=device,
                max_len=max_len,
            )

        with col2:
            st.subheader("Predicted LaTeX")
            st.code(latex_pred, language="latex")

            st.subheader("Rendered formula")
            try:
                st.latex(latex_pred)
            except Exception as e:
                st.error(f"Could not render LaTeX: {e}")
    else:
        st.info("Upload an image to get started.")


if __name__ == "__main__":
    main()
