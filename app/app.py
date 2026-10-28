"""
Streamlit demo: upload an image -> run the full RAG-VLM pipeline -> display
scene description, retrieved facts, final answer, and which method
(primary or fallback) each stage used.
"""

import json
import os
import sys
import tempfile

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Explicitly load .env from project directory
try:
    from dotenv import load_dotenv
    _proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_path = os.path.join(_proj_dir, ".env")
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)
except Exception:
    pass

from pipeline.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG-VLM: Home Safety Scene Assistant", layout="centered")

st.title("🏠 RAG-VLM — Home Safety Hazard Assistant")
st.caption(
    "Upload a photo of a room. The system describes the scene, retrieves "
    "relevant home-safety facts from a knowledge base, and generates a "
    "grounded safety assessment."
)

# ---------------------------------------------------------------------------
# Sidebar: sample images + demo mode toggle
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample_images")
PRECOMPUTED_PATH = os.path.join(DATA_DIR, "precomputed_results.json")

with st.sidebar:
    st.header("Sample images")
    st.caption("No image handy? Try one of the bundled sample scenes.")
    sample_files = sorted(os.listdir(SAMPLE_DIR)) if os.path.isdir(SAMPLE_DIR) else []
    chosen_sample = st.selectbox("Pick a sample", ["(none)"] + sample_files)

    st.divider()
    demo_mode = st.toggle(
        "Demo mode",
        value=False,
        help="Load precomputed results instantly — works offline with no model loading.",
    )

    st.divider()
    st.subheader("⚡ LLM Synthesis (Groq)")
    
    # Try reading from environment, or fallback to reading .env directly if needed
    env_groq_key = os.environ.get("GROQ_API_KEY", "")
    if not env_groq_key:
        try:
            _env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
            if os.path.isfile(_env_file):
                with open(_env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            env_groq_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            os.environ["GROQ_API_KEY"] = env_groq_key
                            break
        except Exception:
            pass

    groq_key_input = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        placeholder="gsk_...",
        help="Key is loaded from .env automatically, or paste a new one here.",
    )

    active_groq_key = groq_key_input.strip() or env_groq_key

    if active_groq_key:
        groq_model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        st.success(f"🟢 Groq API Connected\n\nModel: `{groq_model_name}`")
    else:
        st.warning("⚠️ No Groq API Key set. Using offline template fallback.")


# ---------------------------------------------------------------------------
# Load precomputed results (if the file exists)
# ---------------------------------------------------------------------------
_precomputed: dict = {}
if os.path.isfile(PRECOMPUTED_PATH):
    try:
        with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as f:
            _precomputed = json.load(f)
    except Exception:
        _precomputed = {}

# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

image_path = None
display_image = None
is_sample_image = False

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        image_path = tmp.name
    display_image = uploaded_file
elif chosen_sample != "(none)":
    image_path = os.path.join(SAMPLE_DIR, chosen_sample)
    display_image = image_path
    is_sample_image = True

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if image_path is not None:
    st.image(display_image, caption="Input image", use_container_width=True)

    # -- User question input (hidden in demo mode) --------------------------
    user_question = ""
    if not demo_mode:
        user_question = st.text_input(
            "Ask a question about this image (optional)",
            placeholder="e.g. Is the stove safe for a child? Are there any sharp edge hazards?",
        )

    # -- Method badge helper ------------------------------------------------
    def method_badge(method: str, primary_label: str, fallback_label: str) -> str:
        if method in ("blip2", "vector", "llm", "claude") or str(method).startswith("groq"):
            return f"✅ Used primary method: **{primary_label}**"
        return f"⚠️ Primary method failed — used fallback: **{fallback_label}**"

    # -- Get results (demo or live) -----------------------------------------
    result = None

    if demo_mode and is_sample_image and chosen_sample in _precomputed:
        result = _precomputed[chosen_sample]
        st.info("⚡ Loaded from precomputed results (demo mode)")
    else:
        if demo_mode and is_sample_image:
            st.warning(
                "Precomputed results not found for this image — "
                "falling back to live pipeline."
            )
        with st.spinner("Analyzing scene, retrieving facts, and synthesizing assessment with Groq..."):
            result = run_pipeline(
                image_path,
                user_question=user_question,
                groq_api_key=active_groq_key
            )

    # Clean up temp file from upload
    if uploaded_file is not None:
        try:
            os.unlink(image_path)
        except Exception:
            pass

    # -- Display: Scene Description -----------------------------------------
    with st.container(border=True):
        st.subheader("1. Scene Description")
        st.write(result["description"])
        st.caption(method_badge(result["vision_method"], "BLIP-2", "generic description"))

    # -- Display: Answer based on (explainability) --------------------------
    with st.container(border=True):
        st.subheader("2. Answer based on (Domain Knowledge Base):")
        for fact in result["facts"]:
            st.markdown(f"- {fact}")
        st.caption(method_badge(result["retrieval_method"], "vector search (ChromaDB)", "keyword matching"))

    # -- Display: Final Answer ----------------------------------------------
    with st.container(border=True):
        st.subheader("3. Final Grounded Safety Assessment")
        st.markdown(result["answer"])

        ans_method = result.get("answer_method", "")
        if str(ans_method).startswith("groq"):
            model_used = result.get("answer_model") or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            st.caption(f"✅ Used primary method: **Groq LLM Synthesis ({model_used})**")
        elif ans_method in ("claude", "llm"):
            st.caption("✅ Used primary method: **Claude LLM Synthesis**")
        else:
            st.caption("⚠️ Primary method failed — used fallback: **smart templated summary**")
            st.caption("_(Generated using backup method)_")

    # -- Confidence & Performance Matrix -----------------------------------
    if "metrics" in result:
        metrics = result["metrics"]
        with st.container(border=True):
            st.subheader("4. Safety & Grounding Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Confidence", f"{metrics['overall_confidence']:.1%}")
            col2.metric("Vision Confidence", f"{metrics['vision_confidence']:.1%}")
            col3.metric("Retrieval Similarity", f"{metrics['retrieval_confidence']:.1%}")
            col4.metric("Fact-Checking Score", f"{metrics['fact_checking_score']:.1%}")

    # -- Debug expander -----------------------------------------------------
    with st.expander("Debug info (raw pipeline output)"):
        st.json(result)

else:
    st.info("Upload an image or pick a sample from the sidebar to get started.")
