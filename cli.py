#!/usr/bin/env python3
"""
RAG-VLM Terminal Interface (CLI)

Full-featured terminal application matching all Streamlit capabilities:
- Interactive image picker (bundled samples or custom path/drag-and-drop).
- Demo mode toggle (instant results via precomputed cache, matching Streamlit).
- Live vision model analysis (BLIP-2) with optional safety hazard steering.
- Vector retrieval from home-safety knowledge base (ChromaDB + MiniLM).
- Explainability breakdown: why each fact was retrieved and how it grounds the assessment.
- Full confidence metrics and fact-checking matrix display.
- Optional JSON export for debugging.
"""

import argparse
import glob
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure root dir is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from pipeline.pipeline import run_pipeline

DATA_DIR = os.path.join(ROOT_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample_images")
PRECOMPUTED_PATH = os.path.join(DATA_DIR, "precomputed_results.json")


def load_precomputed() -> dict:
    if os.path.isfile(PRECOMPUTED_PATH):
        try:
            with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_sample_images() -> list:
    if os.path.isdir(SAMPLE_DIR):
        return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
    return []


def print_banner():
    print("\n" + "=" * 78)
    print(" [HOME SAFETY] RAG-VLM: TERMINAL ASSISTANT & HAZARD DETECTOR")
    print("=" * 78)
    print(" Description : Image -> Local BLIP-2 Vision -> ChromaDB Vector KB -> Grounded Assessment")
    print(" Status      : Ready (Supports Live Processing & Offline Demo Mode)")
    print("=" * 78 + "\n")


def interactive_select_image(samples: list) -> tuple:
    """
    Returns (selected_image_path, is_sample, sample_filename)
    """
    print("[Available Bundled Sample Scenes]:")
    for idx, sample_path in enumerate(samples, 1):
        filename = os.path.basename(sample_path)
        print(f"  {idx:2d}) {filename}")
    print("  Or enter a custom file path to any local image (e.g. C:/photos/room.jpg).\n")

    while True:
        choice = input("Enter image path or sample number (1-12) [default: 1]: ").strip()
        # Strip quotes that Windows PowerShell / CMD adds when dragging and dropping files
        choice = choice.strip('"').strip("'")

        if not choice:
            if samples:
                return samples[0], True, os.path.basename(samples[0])
            else:
                print("[!] Please provide an image path.")
                continue

        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(samples):
                chosen = samples[num - 1]
                return chosen, True, os.path.basename(chosen)
            else:
                print(f"[!] Invalid number. Please enter 1 to {len(samples)}.")
                continue

        expanded = os.path.abspath(os.path.expanduser(choice))
        if os.path.isfile(expanded):
            # Check if this matches a bundled sample basename
            base = os.path.basename(expanded)
            is_sample = any(os.path.basename(s) == base for s in samples)
            return expanded, is_sample, base
        else:
            print(f"[!] File not found: '{choice}'. Please verify the path.")


def display_results(result: dict):
    """Prints a clear, structured breakdown matching and exceeding Streamlit's UI."""
    sep = "=" * 78
    thin = "-" * 78

    print("\n" + sep)
    print("  STEP-BY-STEP RAG-VLM ASSESSMENT & EXPLAINABILITY REPORT")
    print(sep)
    print(f"  Image Target : {result['image_path']}")
    if result.get("user_question"):
        print(f"  User Query   : {result['user_question']}")
    print(thin)

    # 1. Vision Scene Description
    vision_method = result.get("vision_method", "unknown")
    v_badge = "[OK] Primary (BLIP-2)" if vision_method == "blip2" else "[WARN] Fallback"
    print(f"  1. SCENE DESCRIPTION ({v_badge}):")
    print(f"     \"{result.get('description', '')}\"")
    if result.get("vision_error"):
        print(f"     Notice: {result['vision_error']}")
    print(thin)

    # 2. Knowledge Base Retrieval & Why it was selected
    retrieval_method = result.get("retrieval_method", "unknown")
    r_badge = "[OK] Primary (ChromaDB + all-MiniLM-L6-v2)" if retrieval_method == "vector" else "[WARN] Fallback Keyword Match"
    print(f"  2. FACTS RETRIEVED FROM KNOWLEDGE BASE ({r_badge}):")
    print(f"     (Ranked by semantic vector relevance to the vision description)")

    metrics = result.get("metrics", {})
    fc_details = metrics.get("fact_checking_details", {})
    fact_details = fc_details.get("fact_details", [])

    if fact_details:
        for item in fact_details:
            idx = item.get("index", 1)
            fact_text = item.get("fact", "")
            score = item.get("score", 0.0)
            citation_type = item.get("citation_type", "Context Only")
            used = item.get("used_in_reasoning", False)

            status_lbl = "[USED IN FINAL REASONING]" if used else "[SUPPORTING CONTEXT]"
            bar = "#" * int(score * 10) + "-" * (10 - int(score * 10))

            print(f"\n     * Fact {idx} [Relevance Score: {score:>5.1%} | {bar}]:")
            print(f"       \"{fact_text}\"")
            print(f"       Grounding in Output : {citation_type} -> {status_lbl}")
    else:
        for i, f in enumerate(result.get("facts", []), 1):
            print(f"     * Fact {i}: {f}")

    print(thin)

    # 3. Grounded Final Answer
    answer_method = result.get("answer_method", "unknown")
    a_badge = "[OK] Primary (LLM)" if answer_method == "llm" else "[FALLBACK] Grounded Synthesis Template"
    print(f"  3. FINAL SAFETY ASSESSMENT ({a_badge}):")
    for line in result.get("answer", "").split("\n"):
        print(f"     {line}")
    print(thin)

    # 4. Performance Matrix & Confidence Scores
    if metrics:
        overall = metrics.get("overall_confidence", 0.0)
        fc_score = metrics.get("fact_checking_score", 0.0)
        v_conf = metrics.get("vision_confidence", 0.0)
        r_conf = metrics.get("retrieval_confidence", 0.0)
        a_conf = metrics.get("answer_confidence", 0.0)

        o_bar = "#" * int(overall * 10) + "-" * (10 - int(overall * 10))
        print("  4. PREDICTED OUTPUT PERFORMANCE & CONFIDENCE MATRIX:")
        print("     +----------------------+----------------+--------+------------+------------+")
        print("     | Pipeline Stage       | Method Used    | Weight | Confidence | Visual Bar |")
        print("     +----------------------+----------------+--------+------------+------------+")
        print(f"     | Vision Understanding | {vision_method:<14} |   25%  | {v_conf:>9.1%}  | [{'#' * int(v_conf * 10) + '-' * (10 - int(v_conf * 10))}] |")
        print(f"     | Knowledge Retrieval  | {retrieval_method:<14} |   25%  | {r_conf:>9.1%}  | [{'#' * int(r_conf * 10) + '-' * (10 - int(r_conf * 10))}] |")
        print(f"     | Answer Synthesis     | {answer_method:<14} |   25%  | {a_conf:>9.1%}  | [{'#' * int(a_conf * 10) + '-' * (10 - int(a_conf * 10))}] |")
        print(f"     | Fact-Checking (KB)   | grounded_check |   25%  | {fc_score:>9.1%}  | [{'#' * int(fc_score * 10) + '-' * (10 - int(fc_score * 10))}] |")
        print("     +----------------------+----------------+--------+------------+------------+")
        print(f"     | OVERALL CONFIDENCE   | Combined       |  100%  | {overall:>9.1%}  | [{o_bar}] |")
        print("     +----------------------+----------------+--------+------------+------------+")

    print(sep + "\n")


def run_cli_session(image_path: str, user_question: str = "", demo_mode: bool = False,
                    is_sample: bool = False, sample_name: str = "", top_k: int = 3,
                    hazard_focus: bool = True, export_json: bool = False):
    precomputed = load_precomputed()

    # Match Streamlit's demo mode behavior
    if demo_mode and is_sample and sample_name in precomputed:
        print(f"\n[DEMO MODE] Loading precomputed instant results for '{sample_name}'...")
        result = precomputed[sample_name]
    else:
        if demo_mode and is_sample:
            print("[INFO] Precomputed cache missed; running live pipeline...")
        print(f"\n>> Executing Live RAG-VLM Pipeline on: {image_path}")
        if user_question:
            print(f"   User Question : \"{user_question}\"")
        if hazard_focus:
            print("   Vision Mode   : Safety & Hazard Detection Prompting Active")

        vision_prompt = "Question: What objects, hazards, or safety concerns are visible in this scene? Answer:" if hazard_focus else ""

        result = run_pipeline(
            image_path=image_path,
            top_k=top_k,
            user_question=user_question,
            vision_prompt=vision_prompt
        )

    display_results(result)

    if export_json:
        out_file = f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[EXPORT] Full raw pipeline output saved to: {out_file}\n")
        except Exception as e:
            print(f"[!] Could not export JSON: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-VLM: Full-featured Terminal Interface for Home Safety Scene Understanding"
    )
    parser.add_argument(
        "-i", "--image", type=str, help="Path to input image file (e.g. data/sample_images/01_kitchen_unattended_stove.png)"
    )
    parser.add_argument(
        "-q", "--question", type=str, default="", help="Optional user question about the scene"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Demo mode: load instant precomputed results for sample images"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of knowledge base facts to retrieve (default: 3)"
    )
    parser.add_argument(
        "--no-hazard-prompt", action="store_true", help="Disable safety-guided vision prompting (use unconditional captioning)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Export full pipeline results to a local JSON file"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run once with provided arguments and exit"
    )

    args = parser.parse_args()

    print_banner()
    samples = get_sample_images()

    if args.image:
        clean_path = args.image.strip('"').strip("'")
        image_path = os.path.abspath(os.path.expanduser(clean_path))
        if not os.path.isfile(image_path):
            print(f"[ERROR] Image file not found: '{args.image}'")
            sys.exit(1)

        base_name = os.path.basename(image_path)
        is_sample = any(os.path.basename(s) == base_name for s in samples)

        run_cli_session(
            image_path=image_path,
            user_question=args.question,
            demo_mode=args.demo,
            is_sample=is_sample,
            sample_name=base_name,
            top_k=args.top_k,
            hazard_focus=not args.no_hazard_prompt,
            export_json=args.json
        )
    else:
        # Interactive Mode
        while True:
            image_path, is_sample, sample_name = interactive_select_image(samples)

            # Interactive options
            q_input = input("\nEnter optional question about this scene (Press Enter to skip): ").strip()

            mode_prompt = input("Run in Demo Mode (instant cached results)? [y/N]: ").strip().lower()
            demo_selected = mode_prompt in ("y", "yes")

            run_cli_session(
                image_path=image_path,
                user_question=q_input,
                demo_mode=demo_selected,
                is_sample=is_sample,
                sample_name=sample_name,
                top_k=args.top_k,
                hazard_focus=not args.no_hazard_prompt,
                export_json=args.json
            )

            if args.non_interactive:
                break

            again = input("Analyze another image in terminal? [Y/n]: ").strip().lower()
            if again in ("n", "no", "exit", "quit"):
                print("\nExiting RAG-VLM Terminal Interface. Goodbye!\n")
                break


if __name__ == "__main__":
    main()

