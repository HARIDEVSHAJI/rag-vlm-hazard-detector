"""
Integration pipeline: image path -> scene description -> retrieved facts ->
grounded final answer.

Primary path : send the scene description + retrieved facts to the Claude
               API and let it synthesize a natural-language answer that
               cites which facts it used.
Fallback path: if the Claude API call fails (no internet, missing/invalid
               API key, rate limit, etc), skip LLM synthesis and return a
               simple templated answer instead. Which path was used is
               always logged in the result.
"""

import os
import re
import sys

# Make sibling packages (vision, retrieval) importable when this file is
# run directly or imported from elsewhere in the project.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from vision.scene_description import describe_scene
from retrieval.knowledge_base import retrieve_facts

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_CLAUDE_MODEL = os.environ.get("RAGVLM_CLAUDE_MODEL", "claude-sonnet-5")


def _call_groq(description: str, facts: list, user_question: str = "", api_key: str = None) -> tuple:
    """
    Calls Groq API with Llama / GPT-OSS models to synthesize a grounded safety assessment.
    Returns (answer_text, model_name).
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or sidebar.")

    from groq import Groq
    client = Groq(api_key=key)

    facts_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    q_line = f'\nUser Inquiry: "{user_question}"\nAddress this specific question in your assessment.' if user_question else ""

    prompt = (
        "You are an expert home safety hazard auditor.\n\n"
        f"Scene Description: {description}\n\n"
        f"Retrieved Home-Safety Facts from Knowledge Base:\n{facts_block}\n"
        f"{q_line}\n\n"
        "Instructions for Assessment:\n"
        "1. Identify and describe what items/areas are visibly present in the scene.\n"
        "2. If hazards or safety concerns exist based on the visible objects (e.g., sharp furniture edges near walkways, slippery floors, tripping hazards), describe them and cite which fact supports each point (e.g., 'as Fact 1 notes...').\n"
        "3. CRITICAL - PREVENT FALSE ALARMS: If any retrieved fact discusses an item that is NOT in this scene (e.g., fireplace, stove, pool, stairs, open flames), EXPLICITLY clarify that this hazard is NOT present or does not apply to this room.\n"
        "4. CLEAN / SAFE SCENES: If the room appears clean, well-arranged, and free of immediate hazards, explicitly state that the scene appears clean and free of immediate safety emergencies, and offer only preventative advice (e.g., non-slip pads for rugs or childproofing sharp table corners if children are present).\n\n"
        "Assessment:"
    )

    models_to_try = [DEFAULT_GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    last_err = None

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            text = response.choices[0].message.content.strip()
            if text:
                text = (
                    text.replace("\u2011", "-")
                    .replace("\u2013", "-")
                    .replace("\u2014", "--")
                    .replace("\u2018", "'")
                    .replace("\u2019", "'")
                    .replace("\u2026", "...")
                )
                return text, model_name

        except Exception as err:
            last_err = err
            continue

    raise RuntimeError(f"Groq API call failed: {last_err}")


def _call_claude(description: str, facts: list, user_question: str = "") -> str:
    """Raises on any failure; caller handles the templated fallback."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    facts_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))

    question_line = ""
    if user_question:
        question_line = (
            f"\n\nThe user also asks: \"{user_question}\"\n"
            "Address their question in your assessment.\n"
        )

    prompt = (
        "You are a home safety assistant. You are given a description of a "
        "room scene and a numbered list of relevant home-safety facts.\n\n"
        "Write a short, grounded assessment of the hazards "
        "visible or implied in the scene. Explicitly reference which fact "
        "number(s) support each point you make (e.g. 'as fact 2 notes...').\n"
        "If a fact mentions an item that is NOT present in this room (like fireplace or stove), "
        "explicitly state that this hazard is not present.\n"
        f"Scene description: {description}\n\n"
        f"Relevant facts:\n{facts_block}\n"
        f"{question_line}\n"
        "Assessment:"
    )

    response = client.messages.create(
        model=DEFAULT_CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()

    if not text:
        raise RuntimeError("Claude API returned an empty response")

    return text


def _templated_answer(description: str, facts: list, user_question: str = "") -> str:
    """
    Synthesize an assessment from the scene description and retrieved facts
    when an external LLM is not available. Explicitly grounds facts and avoids
    claiming absent items are hazards.
    """
    if not facts:
        return f"Assessment for scene ('{description}'): Scene appears clean and no immediate hazards were identified."

    desc_lower = description.lower()
    applicable_points = []
    not_applicable = []

    object_anchors = {
        "fireplace": "fireplace",
        "stove": "stove or kitchen cooking appliances",
        "stairs": "staircase or steps",
        "stair": "staircase or steps",
        "bathtub": "bathtub or shower",
        "pool": "swimming pool",
        "candle": "candles or open flames",
        "space heater": "portable space heaters",
        "heater": "space heaters",
        "garage": "garage or fuel storage",
    }

    for i, f in enumerate(facts, 1):
        f_lower = f.lower()
        absent_item = None
        for anchor_kw, label in object_anchors.items():
            if anchor_kw in f_lower and anchor_kw not in desc_lower:
                absent_item = label
                break

        if absent_item:
            not_applicable.append(f"- Note on Fact {i}: No {absent_item} detected in this room (hazard not present).")
        else:
            applicable_points.append(f"- Observed Consideration (Fact {i}): {f}")

    q_context = f" Regarding your question ('{user_question}'):\n" if user_question else ""

    sections = [f"Hazard Assessment for scene ('{description}'):\n{q_context}"]
    if applicable_points:
        sections.append("Relevant Safety Considerations:\n" + "\n".join(applicable_points))
    else:
        sections.append("Overall Status: Scene appears tidy, well-maintained, and free of immediate safety emergencies.")

    if not_applicable:
        sections.append("Hazards Checked & Verified NOT Present:\n" + "\n".join(not_applicable))

    sections.append("Recommendation: Review the applicable points above for preventative household safety.")
    return "\n\n".join(sections)


def _compute_fact_checking(result: dict) -> dict:
    """
    Computes fact-checking scores by verifying how facts retrieved from the
    knowledge base are grounded in the synthesized answer.
    """
    facts = result.get("facts", [])
    answer = result.get("answer", "").lower()
    scores = result.get("retrieval_scores", [])

    fact_details = []
    cited_count = 0

    for i, fact in enumerate(facts, 1):
        # Check explicit citation references (e.g., "fact 1", "fact #1", "[1]")
        explicit_citation = (
            f"fact {i}" in answer
            or f"fact #{i}" in answer
            or f"[{i}]" in answer
            or f"fact ({i})" in answer
        )

        # Check key-term grounding
        words = [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) > 3]
        matched = [w for w in words if w in answer]
        overlap_ratio = len(matched) / max(len(words), 1)

        is_used = explicit_citation or (overlap_ratio >= 0.25)
        if is_used:
            cited_count += 1

        kb_score = scores[i - 1] if i - 1 < len(scores) else 0.0

        fact_details.append({
            "index": i,
            "fact": fact,
            "score": kb_score,
            "used_in_reasoning": is_used,
            "citation_type": "Explicit Citation" if explicit_citation else ("Keyword Grounded" if is_used else "Context Only")
        })

    grounding_ratio = cited_count / len(facts) if facts else 0.0
    avg_kb_score = sum(scores) / len(scores) if scores else 0.0

    # Fact-checking score combines KB retrieval confidence and answer grounding verification
    fact_checking_score = round(0.5 * avg_kb_score + 0.5 * grounding_ratio, 4)

    return {
        "fact_checking_score": fact_checking_score,
        "grounding_ratio": round(grounding_ratio, 4),
        "cited_count": cited_count,
        "total_facts": len(facts),
        "fact_details": fact_details,
    }


def _compute_confidence(result: dict) -> dict:
    """
    Compute per-stage confidence scores, KB fact-checking score, and overall pipeline confidence.
    """
    vision_conf = 1.0 if result["vision_method"] == "blip2" else 0.2
    scores = result.get("retrieval_scores", [])
    retrieval_conf = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Answer confidence: 1.0 if Groq/Claude succeeded, 0.4 if fallback template
    method = result.get("answer_method", "")
    answer_conf = 1.0 if (method == "llm" or method.startswith("groq") or method == "claude") else 0.4

    fact_check = _compute_fact_checking(result)
    fc_score = fact_check["fact_checking_score"]

    overall = round(0.25 * vision_conf + 0.25 * retrieval_conf + 0.25 * answer_conf + 0.25 * fc_score, 4)

    return {
        "vision_confidence": vision_conf,
        "retrieval_confidence": retrieval_conf,
        "answer_confidence": answer_conf,
        "fact_checking_score": fc_score,
        "overall_confidence": overall,
        "retrieval_fact_scores": scores,
        "fact_checking_details": fact_check,
    }


def _print_performance_matrix(result: dict, metrics: dict):
    """Print a formatted performance matrix to the terminal."""
    sep = "=" * 78
    thin = "-" * 78

    print(f"\n{sep}")
    print("  RAG-VLM: PREDICTED OUTPUT PERFORMANCE & CONFIDENCE MATRIX")
    print(sep)
    print(f"  Input Image  : {result['image_path']}")
    if result.get("user_question"):
        print(f"  User Question: {result['user_question']}")
    print(thin)

    # Performance Matrix Table
    print(f"  +----------------------+----------------+--------+------------+------------+")
    print(f"  | Pipeline Stage       | Method Used    | Weight | Confidence | Visual Bar |")
    print(f"  +----------------------+----------------+--------+------------+------------+")

    fc_details = metrics.get("fact_checking_details", {})
    fc_score = metrics.get("fact_checking_score", 0.0)

    stages = [
        ("Vision Understanding", result["vision_method"],    "25%", metrics["vision_confidence"],   "blip2"),
        ("Knowledge Retrieval",  result["retrieval_method"], "25%", metrics["retrieval_confidence"], "vector"),
        ("Answer Synthesis",     result["answer_method"],    "25%", metrics["answer_confidence"],    "llm"),
        ("Fact-Checking (KB)",   "grounded_check",           "25%", fc_score,                       "grounded_check"),
    ]

    for name, method, weight, conf, primary_key in stages:
        bar = "#" * int(conf * 10) + "-" * (10 - int(conf * 10))
        method_str = method[:14]
        print(f"  | {name:<20} | {method_str:<14} | {weight:>6} | {conf:>9.1%}  | [{bar}] |")

    print(f"  +----------------------+----------------+--------+------------+------------+")
    overall = metrics["overall_confidence"]
    overall_bar = "#" * int(overall * 10) + "-" * (10 - int(overall * 10))
    print(f"  | OVERALL CONFIDENCE   | Combined       |  100%  | {overall:>9.1%}  | [{overall_bar}] |")
    print(f"  +----------------------+----------------+--------+------------+------------+")

    grounded_count = fc_details.get('cited_count', 0)
    total_facts = fc_details.get('total_facts', 0)
    print(f"\n  Fact-Checking Grounding: {grounded_count}/{total_facts} Knowledge Base facts verified & grounded in output.")
    print(thin)

    # Facts Used in Reasoning
    print("  KNOWLEDGE BASE FACTS & EXPLICIT GROUNDING BREAKDOWN:")
    details = fc_details.get("fact_details", [])
    for d in details:
        fact_str = d["fact"]
        short_fact = fact_str[:62] + "..." if len(fact_str) > 62 else fact_str
        score = d["score"]
        bar = "#" * int(score * 10) + "-" * (10 - int(score * 10))
        status_tag = "[USED IN REASONING]" if d["used_in_reasoning"] else "[CONTEXT ONLY]"
        print(f"    Fact {d['index']} | Score: {score:>5.1%} [{bar}] | {d['citation_type']:<16} {status_tag}")
        print(f"           +-- \"{short_fact}\"")

    print(f"{sep}\n")


def run_pipeline(
    image_path: str,
    top_k: int = 3,
    user_question: str = "",
    vision_prompt: str = "",
    groq_api_key: str = None
) -> dict:
    """
    Runs the full image -> description -> facts -> answer pipeline.

    Returns a dict with the outputs of every stage plus which method
    (primary or fallback) each stage actually used, so the UI/tests can
    surface that transparently.
    """
    vision_result = describe_scene(image_path, prompt=vision_prompt)
    description = vision_result["description"]

    # Build the retrieval query: scene description + optional user question.
    retrieval_query = f"{description} {user_question}".strip() if user_question else description

    retrieval_result = retrieve_facts(retrieval_query, top_k=top_k)
    facts = retrieval_result["facts"]

    answer = None
    answer_method = "fallback"
    answer_model = None
    answer_error = None

    # 1. Try Groq first (preferred ultra-fast LLM)
    try:
        answer, answer_model = _call_groq(description, facts, user_question=user_question, api_key=groq_api_key)
        answer_method = f"groq ({answer_model})"
        answer_error = None
    except Exception as groq_err:
        # 2. Try Anthropic Claude if available
        try:
            answer = _call_claude(description, facts, user_question=user_question)
            answer_method = "claude"
            answer_model = DEFAULT_CLAUDE_MODEL
            answer_error = None
        except Exception as claude_err:
            # 3. Graceful offline fallback
            answer = _templated_answer(description, facts, user_question=user_question)
            answer_method = "fallback"
            answer_error = f"Groq: {groq_err} | Claude: {claude_err}"

    result = {
        "image_path": image_path,
        "description": description,
        "user_question": user_question,
        "vision_method": vision_result["method"],
        "vision_error": vision_result["error"],
        "facts": facts,
        "retrieval_scores": retrieval_result.get("scores", []),
        "retrieval_method": retrieval_result["method"],
        "retrieval_error": retrieval_result["error"],
        "answer": answer,
        "answer_method": answer_method,
        "answer_model": answer_model,
        "answer_error": answer_error,
    }

    # Compute and print the performance matrix to terminal.
    metrics = _compute_confidence(result)
    result["metrics"] = metrics
    _print_performance_matrix(result, metrics)

    return result




if __name__ == "__main__":
    import glob

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_images")
    images = sorted(glob.glob(os.path.join(sample_dir, "*.png")))[:3]

    for img_path in images:
        result = run_pipeline(img_path)
        print("=" * 70)
        print(f"Image:       {os.path.basename(img_path)}")
        print(f"Description: {result['description']}  [{result['vision_method']}]")
        print(f"Facts ({result['retrieval_method']}):")
        for f in result["facts"]:
            print(f"  - {f}")
        print(f"Answer [{result['answer_method']}]: {result['answer']}")
