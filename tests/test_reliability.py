"""
Basic reliability test (Step 5).

1. Runs the full pipeline on every sample image in sequence, confirming
   nothing crashes.
2. Deliberately breaks the Claude API path (by clearing ANTHROPIC_API_KEY)
   to confirm the templated-answer fallback actually activates and the
   pipeline still returns a usable answer instead of raising.

Prints a pass/fail summary at the end.
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pipeline import run_pipeline

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_images")


def run_full_dataset():
    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
    print(f"Found {len(images)} sample images.\n")

    results = []
    for img_path in images:
        name = os.path.basename(img_path)
        try:
            result = run_pipeline(img_path)
            ok = bool(result["description"]) and bool(result["answer"])
            results.append((name, ok, result))
            status = "PASS" if ok else "FAIL (empty output)"
            print(
                f"[{status}] {name} | vision={result['vision_method']} "
                f"retrieval={result['retrieval_method']} answer={result['answer_method']}"
            )
        except Exception as e:
            results.append((name, False, None))
            print(f"[FAIL] {name} | CRASHED: {type(e).__name__}: {e}")

    return results


def test_broken_api_key():
    """Deliberately break the API keys to confirm the fallback path activates."""
    print("\n--- Deliberate failure test: broken API keys (Groq & Anthropic) ---")
    orig_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
    orig_groq = os.environ.pop("GROQ_API_KEY", None)
    try:
        images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
        if not images:
            print("[FAIL] no sample images found to test with")
            return False
        result = run_pipeline(images[0])
        fallback_used = result["answer_method"] == "fallback"
        status = "PASS" if fallback_used else "FAIL"
        print(f"[{status}] answer_method={result['answer_method']} (expected 'fallback')")
        print(f"       answer produced: {result['answer']!r}")
        return fallback_used
    except Exception as e:
        print(f"[FAIL] pipeline crashed instead of using fallback: {type(e).__name__}: {e}")
        return False
    finally:
        if orig_anthropic is not None:
            os.environ["ANTHROPIC_API_KEY"] = orig_anthropic
        if orig_groq is not None:
            os.environ["GROQ_API_KEY"] = orig_groq



def main():
    print("=" * 70)
    print("STEP 5 - Basic reliability test")
    print("=" * 70)

    results = run_full_dataset()
    broken_key_ok = test_broken_api_key()

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Full dataset run:      {passed}/{total} images produced a valid answer without crashing")
    print(f"Broken-API-key test:   {'PASS' if broken_key_ok else 'FAIL'} (fallback template activated correctly)")

    overall_pass = (passed == total) and broken_key_ok
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
