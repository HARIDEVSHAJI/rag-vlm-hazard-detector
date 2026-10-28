"""
Retrieval module: query string -> top-k relevant facts.

Primary path : embed facts + query with sentence-transformers
               (all-MiniLM-L6-v2) and search a ChromaDB collection.
Fallback path: plain keyword-overlap scoring between the query and each
               fact, used whenever embedding/vector search fails (no
               internet to fetch the embedding model, ChromaDB error, etc).
"""

import os
import re

os.environ["ANONYMIZED_TELEMETRY"] = "False"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "home_safety_facts"
FACTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "facts.txt")

_collection = None
_embedder = None
_facts_cache = None
_load_failed = False


def _load_facts():
    global _facts_cache
    if _facts_cache is None:
        with open(FACTS_PATH, "r", encoding="utf-8") as f:
            _facts_cache = [line.strip() for line in f if line.strip()]
    return _facts_cache


def _init_vector_store():
    """Lazily build the embedding model + Chroma collection. Raises on failure."""
    global _collection, _embedder, _load_failed
    if _collection is not None:
        return _collection, _embedder
    if _load_failed:
        raise RuntimeError("Vector store initialization previously failed; using keyword fallback.")
    if os.environ.get("RAGVLM_OFFLINE") == "1":
        raise RuntimeError("RAGVLM_OFFLINE environment variable set; using fallback.")

    import socket
    import chromadb
    from sentence_transformers import SentenceTransformer

    try:
        facts = _load_facts()
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception as local_err:
            try:
                s = socket.create_connection(("huggingface.co", 443), timeout=0.5)
                s.close()
            except Exception:
                raise RuntimeError(f"Embedding model not cached & HuggingFace unreachable: {local_err}")
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        embeddings = _embedder.encode(facts).tolist()

        client = chromadb.Client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        collection.add(
            documents=facts,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(facts))],
        )
        _collection = collection
        return _collection, _embedder
    except Exception as e:
        _load_failed = True
        raise e


def _tokenize(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


ANCHOR_OBJECT_CONDITIONS = {
    "fireplace": ["fireplace", "hearth", "chimney"],
    "stove": ["stove", "cook", "oven", "burner", "range", "kitchen", "pot", "pan", "stovetop"],
    "stair": ["stair", "step", "staircase", "steps"],
    "stairs": ["stair", "step", "staircase", "steps"],
    "bathtub": ["bathtub", "shower", "tub", "bathroom", "bath"],
    "shower": ["bathtub", "shower", "tub", "bathroom", "bath"],
    "pool": ["pool", "swimming"],
    "candle": ["candle", "flame"],
    "space heater": ["heater", "radiator"],
    "garage": ["garage", "car", "vehicle"],
    "generator": ["generator"],
    "grill": ["grill", "barbecue", "bbq"],
    "radon": ["radon", "basement"],
    "toilet": ["toilet", "bathroom", "restroom"],
    "dryer": ["dryer", "laundry"],
}


def is_fact_relevant(fact: str, query: str) -> bool:
    """
    Check if a fact requires a specific anchor object (like fireplace, stove, stairs)
    and whether that object is mentioned in the query.
    If a fact mentions 'fireplace', and the query doesn't mention fireplace/hearth,
    it is filtered out so it doesn't crowd out actual relevant facts.
    """
    query_lower = query.lower()
    fact_lower = fact.lower()
    for anchor, triggers in ANCHOR_OBJECT_CONDITIONS.items():
        if re.search(r"\b" + re.escape(anchor) + r"\b", fact_lower):
            has_trigger = any(re.search(r"\b" + re.escape(t) + r"\b", query_lower) for t in triggers)
            if not has_trigger:
                return False
    return True


def _keyword_fallback(query: str, top_k: int):
    """Simple, dependency-free keyword-overlap retrieval. Returns (facts, scores)."""
    facts = _load_facts()
    query_words = _tokenize(query)

    scored = []
    for fact in facts:
        fact_words = _tokenize(fact)
        overlap = len(query_words & fact_words)
        # Normalize by the size of the smaller set to get a 0-1 score.
        denom = min(len(query_words), len(fact_words)) if query_words and fact_words else 1
        score = overlap / max(denom, 1)
        scored.append((score, fact))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter by anchor relevance first
    relevant_scored = [(s, f) for s, f in scored if is_fact_relevant(f, query)]
    remaining_scored = [(s, f) for s, f in scored if not is_fact_relevant(f, query)]
    final_scored = (relevant_scored + remaining_scored)[:top_k]

    top_facts = [fact for score, fact in final_scored]
    top_scores = [round(score, 4) for score, fact in final_scored]
    return top_facts, top_scores


def retrieve_facts(query: str, top_k: int = 3) -> dict:
    """
    Takes a query string, returns the top-k relevant facts.

    Returns:
        {
            "facts": [str, ...],
            "scores": [float, ...],   # similarity scores (0-1, higher = better)
            "method": "vector" | "fallback",
            "error": str | None,
        }
    """
    try:
        collection, embedder = _init_vector_store()
        query_embedding = embedder.encode([query]).tolist()
        
        # Query more candidates so we can filter out inapplicable anchors (e.g. fireplace in living room)
        n_candidates = min(top_k * 5, len(_load_facts()))
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_candidates,
            include=["documents", "distances"],
        )
        all_facts = results["documents"][0]
        distances = results["distances"][0]

        if not all_facts:
            raise ValueError("Vector search returned no results")

        all_scores = [round(1.0 / (1.0 + d), 4) for d in distances]
        
        # Separate into relevant anchor facts and inapplicable ones
        relevant = []
        inapplicable = []
        for fact, score in zip(all_facts, all_scores):
            if is_fact_relevant(fact, query):
                relevant.append((fact, score))
            else:
                inapplicable.append((fact, score))

        # Take up to top_k relevant facts; fallback to others only if not enough relevant ones
        selected = (relevant + inapplicable)[:top_k]
        facts = [f for f, s in selected]
        scores = [s for f, s in selected]

        return {"facts": facts, "scores": scores, "method": "vector", "error": None}

    except Exception as e:
        facts, scores = _keyword_fallback(query, top_k)
        return {
            "facts": facts,
            "scores": scores,
            "method": "fallback",
            "error": f"{type(e).__name__}: {e}",
        }



if __name__ == "__main__":
    test_queries = [
        "a kitchen with a pot boiling unattended on the stove",
        "a staircase with toys and shoes scattered on the steps",
        "a bathroom with a wet slippery floor near the tub",
    ]

    for q in test_queries:
        result = retrieve_facts(q)
        print(f"\nQuery: {q}")
        print(f"Method: {result['method']}" + (f" (reason: {result['error']})" if result["error"] else ""))
        for fact in result["facts"]:
            print(f"  - {fact}")
