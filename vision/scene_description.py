"""
Vision module: image -> scene description.

Primary path : BLIP-2 (Salesforce/blip2-opt-2.7b) via Hugging Face transformers.
Fallback path: a generic scene description, used whenever the model can't be
               loaded or fails on a given image (no internet, OOM, corrupt
               image, etc). The pipeline must never crash because of this
               module.
"""

import os
import socket

from PIL import Image

MODEL_NAME = "Salesforce/blip2-opt-2.7b"
FALLBACK_DESCRIPTION = "An indoor/outdoor scene with various objects"

_model = None
_processor = None
_load_failed = False


def _load_model():
    """Lazily load BLIP-2. Raises on failure; caller handles fallback."""
    global _model, _processor, _load_failed

    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    if _model is not None:
        return _model, _processor
    if _load_failed:
        raise RuntimeError("BLIP-2 model load failed on prior attempt; using fallback.")
    if os.environ.get("RAGVLM_OFFLINE") == "1":
        raise RuntimeError("RAGVLM_OFFLINE environment variable set; using fallback.")

    try:
        try:
            processor = Blip2Processor.from_pretrained(MODEL_NAME, local_files_only=True)
            model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception as local_err:
            # Fast check if huggingface.co is reachable (0.5s timeout)
            try:
                s = socket.create_connection(("huggingface.co", 443), timeout=0.5)
                s.close()
            except Exception:
                raise RuntimeError(f"BLIP-2 local model not cached & HuggingFace unreachable: {local_err}")

            processor = Blip2Processor.from_pretrained(MODEL_NAME)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        _model, _processor = model, processor
        return _model, _processor
    except Exception as e:
        _load_failed = True
        raise e


def describe_scene(image_path: str, prompt: str = "") -> dict:
    """
    Takes an image path and optional text prompt, returns a scene description.

    Returns:
        {
            "description": str,
            "method": "blip2" | "fallback",
            "error": str | None,   # populated when the fallback was used
        }
    """
    try:
        model, processor = _load_model()
        image = Image.open(image_path).convert("RGB")
        if prompt:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        else:
            inputs = processor(images=image, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if not description:
            raise ValueError("BLIP-2 returned an empty description")

        return {"description": description, "method": "blip2", "error": None}

    except Exception as e:
        return {
            "description": FALLBACK_DESCRIPTION,
            "method": "fallback",
            "error": f"{type(e).__name__}: {e}",
        }


if __name__ == "__main__":
    # Quick manual test on a couple of sample images.
    import os
    import glob

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_images")
    images = sorted(glob.glob(os.path.join(sample_dir, "*.png")))[:3]

    for img_path in images:
        result = describe_scene(img_path)
        print(f"[{result['method']}] {os.path.basename(img_path)} -> {result['description']}")
        if result["error"]:
            print(f"    (reason for fallback: {result['error']})")
