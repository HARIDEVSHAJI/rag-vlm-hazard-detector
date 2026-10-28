# RAG-VLM: Home Safety Hazard Assistant
## Comprehensive Technical Project Documentation

---

## 1. Introduction & Problem Statement

**RAG-VLM** (Retrieval-Augmented Vision-Language Model) is a robust, multimodal AI architecture designed for domain-specific scene understanding—specifically, detecting home safety hazards. 

**The Core Problem:**
1. **Context Blindness in LLMs:** Standard LLMs cannot process visual information natively without a vision encoder.
2. **Hallucination in VLMs:** While generalized Vision-Language Models (VLMs) can describe images, they often lack strict domain-specific knowledge (like precise safety regulations) and are prone to hallucinations when asked for compliance assessments.
3. **Pipeline Fragility:** Multi-stage AI pipelines (Vision → Embedding → Vector DB → LLM) typically suffer from a "weakest link" problem. If the internet drops, an API key is missing, or a massive 15GB model fails to load into VRAM, the entire application crashes.

**The Solution:**
RAG-VLM solves this by decoupling the architecture into three strict components: Vision (seeing), Retrieval (knowing), and Synthesis (reasoning). Crucially, it wraps every single component in a **Zero-Crash Graceful Degradation** framework, ensuring that the pipeline always returns a functional response regardless of hardware or network constraints.

---

## 2. Explicit Feature List

* **Multimodal RAG Pipeline:** Combines computer vision (BLIP-2) with text-based Retrieval-Augmented Generation to ground image analysis in actual documented facts.
* **Zero-Crash Graceful Degradation (Fallbacks):** Every single module (Vision, Retrieval, LLM) has a hardcoded, offline-capable fallback mechanism. The app will never throw a raw exception to the user.
* **Interactive User Question Support:** Users can optionally input specific questions (e.g., "Is this safe for a toddler?"). The system dynamically combines the scene description and the user query to steer the vector search and the final LLM prompt.
* **Transparent Explainability UI:** The Streamlit interface explicitly exposes the pipeline's internal state. It displays exactly which facts were retrieved ("Answer based on:") and includes dynamic UI badges (✅ / ⚠️) indicating whether a primary AI model or a backup fallback was triggered for each stage.
* **Offline Demo Mode:** A precomputation script allows the entire pipeline to be run in advance on sample images, caching the results to a local JSON file. A UI toggle allows users to instantly load these precomputed results, demonstrating the UI and data flow with zero latency and absolutely no active network/GPU requirements.

---

## 3. Technology Stack & Models

### Core Libraries
* **Language:** Python 3
* **Application UI:** `streamlit`
* **Machine Learning / Tensors:** `torch` (PyTorch)
* **NLP & Vision Transformers:** `transformers` (Hugging Face)
* **Vector Embeddings:** `sentence-transformers`
* **Vector Database:** `chromadb`
* **LLM API:** `anthropic` (Claude API)
* **Image Processing:** `Pillow` (PIL)

### AI Models Utilized
1. **Vision Encoder:** `Salesforce/blip2-opt-2.7b`
   * *Role:* Image-to-text captioning. Generates a strict, literal description of the scene.
2. **Embedding Model:** `all-MiniLM-L6-v2`
   * *Role:* Maps sentences and paragraphs to a 384-dimensional dense vector space for semantic similarity search.
3. **Large Language Model:** `claude-sonnet-5` (Anthropic)
   * *Role:* Synthesizes the visual description and the retrieved domain knowledge into a natural language assessment.

---

## 4. Deep-Dive: System Architecture & Workflow

The system is highly modular. Below is the technical breakdown of data flow and fallback handling across the pipeline.

### Stage 1: Vision Processing (`vision/scene_description.py`)
* **Primary Path:** The module lazily loads `Blip2Processor` and `Blip2ForConditionalGeneration`. It intelligently checks for hardware acceleration (`torch.cuda.is_available()`), casting to `torch.float16` if a GPU is present to save VRAM, or `torch.float32` for CPU execution. The image is passed through the processor, and `model.generate(max_new_tokens=40)` creates a concise text description.
* **Fallback Path:** If the `transformers` library is missing, the weights fail to download, or an Out-Of-Memory (OOM) error occurs, the `try/except` block catches the exception and returns a hardcoded generic description: *"An indoor/outdoor scene with various objects"*.

### Stage 2: Knowledge Retrieval (`retrieval/knowledge_base.py`)
* **Knowledge Base:** A local text file (`data/facts.txt`) containing 53 specific home-safety rules.
* **Primary Path:** The script utilizes ChromaDB as an ephemeral, in-memory vector store. It loads the `all-MiniLM-L6-v2` SentenceTransformer model to encode the 53 facts into 384-dimensional embeddings. At inference, it takes the scene description (and appends the `user_question` if provided), embeds this query, and performs a Cosine Similarity search via ChromaDB to return the `top_k` (default: 3) most relevant facts.
* **Fallback Path:** If ChromaDB or the embedding model fails (e.g., offline environment without cached Hugging Face weights), the system falls back to a custom **Jaccard-style Keyword Overlap algorithm**. It tokenizes the query and the facts using regex (`re.findall(r"[a-z0-9]+")`), calculates set intersection sizes, and sorts the facts by highest keyword overlap.

### Stage 3: LLM Synthesis (`pipeline/pipeline.py`)
* **Primary Path:** The pipeline compiles a strict system prompt. It combines:
  1. The task instruction (act as a safety assistant, only use provided facts).
  2. `description`: The output from Stage 1.
  3. `facts`: A numbered list of strings from Stage 2.
  4. `user_question`: Injected instruction to address the user's specific concern, if provided.
  The prompt is sent to the Anthropic API (`messages.create` with `max_tokens=300`).
* **Fallback Path:** If `ANTHROPIC_API_KEY` is missing from the environment, or a network/rate-limit error occurs, the pipeline bypasses the LLM and executes a string-formatting template: `f"Scene: {description}. Relevant considerations: {facts_list}."`

### Stage 4: Interactive Frontend (`app/app.py`)
* **State Management:** Streamlit handles file uploads via `st.file_uploader`, writing the binary data to a secure temporary file (`tempfile.NamedTemporaryFile`) so the pipeline can read it from a standard file path, unlinking (deleting) the temp file immediately after analysis.
* **Demo Mode Routing:** If "Demo Mode" is enabled via `st.toggle` and a sample image is selected, the app bypasses `run_pipeline()` entirely. It reads a precomputed JSON dictionary (`data/precomputed_results.json`) and loads the exact payload in milliseconds.
* **Explainability Rendering:** The UI utilizes `st.container(border=True)` to create distinct output blocks. A custom `method_badge` function reads the `vision_method`, `retrieval_method`, and `answer_method` keys from the pipeline dictionary to render Markdown badges (e.g., `✅ Used primary method: BLIP-2` or `⚠️ Primary method failed — used fallback: templated summary`).

---

## 5. Project Structure

```text
rag-vlm/
├── app/
│   └── app.py                     # Streamlit UI, state management, and routing
├── data/
│   ├── facts.txt                  # The text-based Knowledge Base (53 facts)
│   ├── sample_images/             # 12 synthetic standard test images
│   ├── precompute_results.py      # Batch-processing script to generate JSON cache
│   └── precomputed_results.json   # Cached pipeline outputs for Demo Mode
├── pipeline/
│   └── pipeline.py                # Pipeline orchestrator & Claude API integrations
├── retrieval/
│   └── knowledge_base.py          # ChromaDB, SentenceTransformers, and keyword fallback
├── vision/
│   └── scene_description.py       # Hugging Face BLIP-2 lazy loading and generic fallback
├── tests/
│   └── test_reliability.py        # Automated test suite (asserts fallback behavior)
├── docs/
│   └── PROJECT_DOCUMENTATION.md   # This document
└── requirements.txt               # Strict dependency versions
```

## 6. Running the Project

**1. Install Dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure Environment (Optional but recommended):**
```bash
export ANTHROPIC_API_KEY="your_claude_api_key"
```

**3. Launch the Application:**
```bash
streamlit run app/app.py
```

**4. Run Reliability Tests:**
Ensures all fallback paths execute properly across the 12 sample images:
```bash
python tests/test_reliability.py
```
