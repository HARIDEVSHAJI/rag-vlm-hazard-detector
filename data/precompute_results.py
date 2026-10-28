"""
Precompute pipeline results for all sample images.

Runs the full RAG-VLM pipeline on every .png in data/sample_images/ and
saves the results to data/precomputed_results.json, keyed by filename.

Usage:
    HF_HUB_OFFLINE=1 python data/precompute_results.py
"""

import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pipeline import run_pipeline

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "precomputed_results.json")


def main():
    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
    print(f"Found {len(images)} sample images in {SAMPLE_DIR}")

    results = {}
    for img_path in images:
        name = os.path.basename(img_path)
        print(f"  Processing {name} ... ", end="", flush=True)
        try:
            result = run_pipeline(img_path)
            # Store the filename as the image_path key (portable).
            result["image_path"] = name
            results[name] = result
            print(f"OK [{result['vision_method']}/{result['retrieval_method']}/{result['answer_method']}]")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
