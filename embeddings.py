# embeddings.py — MemoryOS Day 4
# Real TF-IDF Embeddings (No torch needed!)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime

# ── REAL EMBEDDING ENGINE ────────────────────────────
class RealEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.is_fitted = False

    def fit(self, texts: list):
        """Vocabulary seekho"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"✅ Embedder trained on {len(texts)} texts")
        print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def embed(self, texts: list) -> np.ndarray:
        """Text ko vector mein convert karo"""
        if not self.is_fitted:
            self.fit(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        return np.mean(vectors, axis=0)

# ── DRIFT SCORE ──────────────────────────────────────
def drift_score(v1: np.ndarray, v2: np.ndarray) -> float:
    sim = cosine_similarity([v1], [v2])[0][0]
    return round(1.0 - float(sim), 4)

def get_status(score: float) -> str:
    if score < 0.3:   return "🟢 HEALTHY"
    elif score < 0.5: return "🟡 WARNING"
    elif score < 0.7: return "🔴 CRITICAL"
    else:             return "🚨 DANGER"

# ── MAIN TEST ────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 MemoryOS — Real Embeddings (TF-IDF)")
    print("="*45)

    embedder = RealEmbedder()

    # Baseline
    print("\n📍 Creating baseline...")
    baseline_texts = [
        "what is machine learning",
        "what is deep learning",
        "what is neural network",
        "what is python programming",
        "explain artificial intelligence"
    ]
    embedder.fit(baseline_texts)
    baseline_vector = embedder.embed(baseline_texts)
    print(f"   Vector dimensions: {baseline_vector.shape[0]}")

    # Test 1 — Similar
    print("\n🔍 Test 1: Similar AI topics...")
    similar = ["supervised learning tutorial",
               "how neural networks work",
               "deep learning basics"]
    v1 = embedder.embed(similar)
    s1 = drift_score(baseline_vector, v1)
    print(f"   Drift Score: {s1} — {get_status(s1)}")

    # Test 2 — Different
    print("\n🔍 Test 2: Completely different...")
    different = ["best cricket team world",
                 "butter chicken recipe",
                 "bollywood movies 2026"]
    v2 = embedder.embed(different)
    s2 = drift_score(baseline_vector, v2)
    print(f"   Drift Score: {s2} — {get_status(s2)}")

    # Test 3 — Same
    print("\n🔍 Test 3: Almost same topics...")
    same = ["machine learning python",
            "deep learning neural network",
            "AI programming tutorial"]
    v3 = embedder.embed(same)
    s3 = drift_score(baseline_vector, v3)
    print(f"   Drift Score: {s3} — {get_status(s3)}")

    print("\n" + "="*45)
    print("📊 Results Summary:")
    print(f"   Similar topics:   {s1} {get_status(s1)}")
    print(f"   Different topics: {s2} {get_status(s2)}")
    print(f"   Same topics:      {s3} {get_status(s3)}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "TF-IDF (scikit-learn)",
        "vector_dimensions": int(baseline_vector.shape[0]),
        "results": {
            "similar_drift": s1,
            "different_drift": s2,
            "same_drift": s3
        }
    }
    with open("embedding_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n💾 Results saved!")
    print("✅ Day 4 Complete!")