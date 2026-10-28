# RAG-VLM: Multimodal Retrieval-Augmented Generation for Home Safety Hazard Detection
## System Architecture, Technical Specification & AI Assistant Handoff Guide

> **Note for the incoming AI Model / Assistant:**
> You are working on the **RAG-VLM** codebase. Read this document completely to understand the project architecture, data flow, failover design, file roles, and coding conventions. You are equipped to answer questions, refactor, add features, optimize, or modify code in **any format requested by the user** (e.g., unified diffs, whole-file replacements, step-by-step walkthroughs, or architectural proposals).

---

## 1. Project Overview & Problem Statement

**RAG-VLM** is a production-grade, multimodal AI pipeline designed for **Home Safety Hazard Assessment**. Given an image of a residential room or area (e.g., kitchen with unattended stove, bathroom with water puddle, cluttered stairs, exposed wiring), the system:
1. **Sees & Describes**: Extracts a factual scene description from the image.
2. **Retrieves Domain Knowledge**: Searches a verified home safety knowledge base for relevant safety regulations, hazards, and guidelines.
3. **Synthesizes Grounded Assessment**: Generates an authoritative hazard evaluation citing specific retrieved facts.
4. **Verifies & Scores**: Computes explicit citation grounding and a multi-stage confidence matrix.

### Core Engineering Principle: Zero-Crash Graceful Degradation
Every external AI component (Hugging Face model download, PyTorch inference, ChromaDB vector store, Anthropic Claude API) is wrapped in resilient `try/except` fallback logic. **The application never crashes with an unhandled exception** due to missing hardware (no CUDA GPU), missing packages, lack of internet, or absent API keys. It transparently falls back to heuristic/offline algorithms and explicitly tags which method was used at each step.

---

## 2. High-Level Architecture & Data Flow

```
                      +-----------------------------+
                      |         Input Image         |
                      |  (Local File / User Upload) |
                      +--------------+--------------+
                                     |
                                     v
                 +---------------------------------------+
                 |       Stage 1: Vision Module          |
                 |      `vision/scene_description.py`    |
                 +-------------------+-------------------+
                                     |
             +-----------------------+-----------------------+
             | Primary Path                                  | Fallback Path
             v                                               v
    [BLIP-2 Vision Model]                           [Generic Scene Caption]
 (Salesforce/blip2-opt-2.7b)                       ("An indoor/outdoor scene...")
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v  `description` + optional `user_question`
                 +---------------------------------------+
                 |      Stage 2: Retrieval Module        |
                 |     `retrieval/knowledge_base.py`     |
                 +-------------------+-------------------+
                                     |
             +-----------------------+-----------------------+
             | Primary Path                                  | Fallback Path
             v                                               v
   [MiniLM + ChromaDB Vector DB]                   [Jaccard Keyword Overlap]
   (all-MiniLM-L6-v2 Embeddings)                   (Token set overlap on facts)
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v  Top-K Retrieved `facts` + `scores`
                 +---------------------------------------+
                 |      Stage 3: Synthesis Module        |
                 |         `pipeline/pipeline.py`        |
                 +-------------------+-------------------+
                                     |
             +-----------------------+-----------------------+
             | Primary Path                                  | Fallback Path
             v                                               v
   [Claude 3.5 / Sonnet LLM]                       [Rule-Based Template Engine]
   (Anthropic API Messages SDK)                    (Structured citation bullets)
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                 +---------------------------------------+
                 |    Stage 4: Grounding & Evaluation    |
                 |         `pipeline/pipeline.py`        |
                 +-------------------+-------------------+
                                     |
             +-----------------------+-----------------------+
             | Citation Grounding Ratio Check (Regex/Tokens) |
             | Multi-Stage Performance & Confidence Matrix   |
             +-----------------------+-----------------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
    [Terminal CLI Interface]                       [Streamlit Web UI]
          (`cli.py`)                                 (`app/app.py`)
```

---

## 3. Directory & File Structure

```text
rag-vlm/
│
├── vision/
│   └── scene_description.py      # Stage 1: BLIP-2 vision-to-language with generic caption fallback
│
├── retrieval/
│   └── knowledge_base.py         # Stage 2: ChromaDB + MiniLM semantic vector search with keyword fallback
│
├── pipeline/
│   └── pipeline.py               # Stage 3 & 4: Pipeline orchestrator, Claude synthesis, confidence matrix
│
├── app/
│   └── app.py                    # Stage 5: Interactive Streamlit web application & demo mode
│
├── cli.py                        # Full-featured terminal interface matching all Streamlit capabilities
│
├── data/
│   ├── facts.txt                 # Domain Knowledge Base (28 curated home safety hazard facts)
│   ├── sample_images/            # 12 synthetic room scene images (kitchen, stairs, bathroom, etc.)
│   ├── generate_sample_images.py # PIL script to regenerate synthetic test scenes
│   ├── precompute_results.py     # Script to generate precomputed cache for instant zero-latency demo
│   └── precomputed_results.json  # Cached pipeline results for all 12 sample images
│
├── tests/
│   ├── test_reliability.py       # Automated test suite validating fallback activation and stability
│   └── sample_run_log.txt        # Verified output logs from sandbox execution
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md  # Comprehensive technical project documentation
│
└── requirements.txt              # Project dependencies (torch, transformers, chromadb, anthropic, etc.)
```

---

## 4. Detailed Component Specifications

### 4.1 Vision Module (`vision/scene_description.py`)
* **Function**: `describe_scene(image_path: str, prompt: str = "") -> dict`
* **Primary Path**:
  * Loads `Salesforce/blip2-opt-2.7b` via Hugging Face `transformers`.
  * Auto-detects CUDA: runs `torch.float16` on GPU, `torch.float32` on CPU.
  * Generates descriptive scene tokens (`max_new_tokens=50`).
* **Fallback Path**:
  * Triggers on `ImportError`, network failure, Hugging Face Hub unreachable, CUDA OOM, or missing weights.
  * Returns: `"An indoor/outdoor scene with various objects"`
* **Output Schema**:
  ```python
  {
      "description": str,
      "method": "blip2" | "fallback",
      "error": str | None
  }
  ```

### 4.2 Retrieval Module (`retrieval/knowledge_base.py`)
* **Function**: `retrieve_facts(query: str, top_k: int = 3) -> dict`
* **Knowledge Source**: `data/facts.txt` (28 domain safety statements).
* **Primary Path**:
  * Embeds facts using `all-MiniLM-L6-v2` (`sentence-transformers`).
  * In-memory `chromadb.Client()` collection: queries using cosine similarity.
  * Returns top $k$ facts and normalized similarity scores.
* **Fallback Path**:
  * Triggers if ChromaDB or `sentence-transformers` is missing or offline.
  * Uses token set intersection (Jaccard-like overlap normalized by $\min(|Q|, |F|)$).
* **Output Schema**:
  ```python
  {
      "facts": list[str],
      "scores": list[float],
      "method": "vector" | "fallback",
      "error": str | None
  }
  ```

### 4.3 Pipeline & Synthesis (`pipeline/pipeline.py`)
* **Function**: `run_pipeline(image_path: str, top_k: int = 3, user_question: str = "", vision_prompt: str = "") -> dict`
* **Data Flow**:
  1. Calls `describe_scene(image_path, prompt=vision_prompt)`.
  2. Constructs retrieval query: `f"{description} {user_question}".strip()`.
  3. Calls `retrieve_facts(retrieval_query, top_k=top_k)`.
  4. Synthesizes final grounded assessment:
     * **Primary Path**: `_call_claude(description, facts, user_question)` using Anthropic SDK (`claude-sonnet-5`).
     * **Fallback Path**: `_templated_answer(description, facts, user_question)` building structured citations.
  5. Computes fact-checking grounding & confidence metrics via `_compute_confidence()`.
* **Output Schema**:
  ```python
  {
      "image_path": str,
      "user_question": str,
      "description": str,
      "facts": list[str],
      "retrieval_scores": list[float],
      "answer": str,
      "vision_method": "blip2" | "fallback",
      "retrieval_method": "vector" | "fallback",
      "answer_method": "llm" | "fallback",
      "metrics": {
          "vision_confidence": float,       # 1.0 (blip2) or 0.2 (fallback)
          "retrieval_confidence": float,    # Average retrieval similarity score
          "answer_confidence": float,       # 1.0 (llm) or 0.4 (fallback)
          "fact_checking_score": float,     # Grounding verification score
          "overall_confidence": float,      # Weighted 4-part average (25% each)
          "retrieval_fact_scores": list[float],
          "fact_checking_details": {
              "fact_checking_score": float,
              "grounding_ratio": float,
              "cited_count": int,
              "total_facts": int,
              "fact_details": list[dict]    # Breakdown per fact
          }
      }
  }
  ```

### 4.4 Fact-Checking & Confidence Metric Formulas
* **Grounding Check**: Evaluates whether retrieved facts appear in the synthesized answer through:
  * Explicit citation patterns (`"fact 1"`, `"fact #1"`, `"[1]"`).
  * Keyword lemma overlap ($\ge 25\%$ significant word match).
* **Overall Confidence Weighting**:
  $$\text{Overall Confidence} = 0.25 \times C_{\text{vision}} + 0.25 \times C_{\text{retrieval}} + 0.25 \times C_{\text{answer}} + 0.25 \times S_{\text{fact-checking}}$$

---

## 5. User Interfaces & Execution

### 5.1 Terminal CLI (`cli.py`)
Run the interactive terminal interface:
```bash
# Interactive mode with image selector & prompt
python cli.py

# Direct image evaluation
python cli.py --image data/sample_images/01_kitchen_unattended_stove.png

# User inquiry steering
python cli.py --image data/sample_images/01_kitchen_unattended_stove.png --question "Is this safe for toddlers?"

# Instant demo mode (offline precomputed cache)
python cli.py --image data/sample_images/01_kitchen_unattended_stove.png --demo

# Raw JSON output (for scripting/API piping)
python cli.py --image data/sample_images/01_kitchen_unattended_stove.png --json
```

### 5.2 Streamlit Web Application (`app/app.py`)
Launch the browser dashboard:
```bash
streamlit run app/app.py
```
Features:
* Image uploader (drag-and-drop JPEG/PNG) and bundled sample selector.
* Interactive question input.
* Demo mode toggle (loads `data/precomputed_results.json` instantly without GPU/network).
* Visual badges indicating primary vs fallback paths (`✅` / `⚠️`).
* Grounding verification and fact breakdown table.

### 5.3 Automated Tests (`tests/test_reliability.py`)
```bash
python tests/test_reliability.py
```
Validates that all 12 sample images process cleanly through the pipeline with zero unhandled exceptions, and asserts that fallback paths function as intended under simulated failure conditions.

---

## 6. Environment Variables

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `ANTHROPIC_API_KEY` | API Key for Claude LLM synthesis | `None` (triggers templated fallback) |
| `RAGVLM_CLAUDE_MODEL` | Claude model variant identifier | `claude-sonnet-5` |
| `RAGVLM_OFFLINE` | Force offline mode (skip Hugging Face network attempts) | `0` (or `1` to force fallback) |
| `ANONYMIZED_TELEMETRY` | ChromaDB telemetry suppression | `"False"` |

---

## 7. Instructions for Incoming AI Models (How to Edit Code)

When the user asks you to modify, enhance, or refactor this codebase, follow these rules:

1. **Maintain Zero-Crash Guarantees**:
   * Never remove `try/except` fallback blocks in `vision/scene_description.py`, `retrieval/knowledge_base.py`, or `pipeline/pipeline.py`.
   * If introducing a new dependency (e.g., OpenCV, alternative VLM, new vector store, OpenAI API), ensure a graceful offline fallback remains operational if that dependency is unavailable.

2. **Preserve the Return Schema**:
   * Functions like `describe_scene`, `retrieve_facts`, and `run_pipeline` return dictionaries with established keys (`description`, `method`, `error`, `facts`, `answer`, `metrics`).
   * When adding new metadata, add new keys without deleting or renaming existing keys to prevent breaking `cli.py`, `app/app.py`, and `test_reliability.py`.

3. **Format Adaptability**:
   * The user may ask for edits in **any format**. You are authorized to output:
     * Standard unified diffs (`diff` blocks).
     * Complete drop-in replacement files.
     * Targeted search-and-replace blocks.
     * Architectural guides or explanatory walkthroughs.
   * If the user does not specify a format, provide clean, concise code with clear explanations.

4. **Multi-File Synchronization**:
   * If you add new parameters to `run_pipeline` (in `pipeline/pipeline.py`), remember to expose them in both `cli.py` (command-line argument) and `app/app.py` (Streamlit UI widget).

5. **Path Handling on Windows**:
   * Use `os.path.join` and `os.path.abspath` for file paths.
   * Strip surrounding quotes when accepting user input in `cli.py` to support drag-and-drop file paths on Windows.
