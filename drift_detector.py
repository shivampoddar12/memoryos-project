# drift_detector.py — MemoryOS Week 1, Day 2
# Drift Detection Engine — cosine similarity se

import numpy as np
import json
from datetime import datetime

# ── SIMPLE EMBEDDING ─────────────────────────────────
# (Abhi simple word-count embedding use karenge
#  Week 3 mein real sentence-transformer add karenge)

def simple_embed(text: str, vocab_size: int = 50) -> np.ndarray:
    """Text ko number vector mein convert karo"""
    words = text.lower().split()
    vector = np.zeros(vocab_size)
    for word in words:
        # Har word ka index calculate karo
        idx = hash(word) % vocab_size
        vector[idx] += 1
    # Normalize karo
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

# ── COSINE SIMILARITY ────────────────────────────────
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Do vectors kitne similar hain — 1.0 = same, 0.0 = bilkul alag"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# ── DRIFT SCORE ──────────────────────────────────────
def calculate_drift(baseline_vector: np.ndarray,
                    current_vector: np.ndarray) -> float:
    """
    drift = 1 - cosine_similarity(baseline, current)
    0.0 = no drift (perfect)
    1.0 = complete drift (agent gone rogue!)
    """
    similarity = cosine_similarity(baseline_vector, current_vector)
    drift = 1.0 - similarity
    return round(drift, 4)

# ── DRIFT STATUS ─────────────────────────────────────
def get_drift_status(drift_score: float) -> dict:
    """Drift score ka matlab kya hai"""
    if drift_score < 0.3:
        return {
            "status": "HEALTHY",
            "emoji": "🟢",
            "message": "Agent stable hai — no action needed"
        }
    elif drift_score < 0.5:
        return {
            "status": "WARNING",
            "emoji": "🟡",
            "message": "Thoda drift detect hua — monitor karo"
        }
    elif drift_score < 0.7:
        return {
            "status": "CRITICAL",
            "emoji": "🔴",
            "message": "High drift! Auto-heal soon needed"
        }
    else:
        return {
            "status": "DANGER",
            "emoji": "🚨",
            "message": "ALERT! Agent severely drifted — heal NOW!"
        }

# ── DRIFT ENGINE CLASS ───────────────────────────────
class DriftDetector:
    def __init__(self):
        self.baseline_vector = None
        self.drift_history = []
        self.session_count = 0

    def set_baseline(self, texts: list):
        """Pehle session se baseline capture karo"""
        vectors = [simple_embed(t) for t in texts]
        self.baseline_vector = np.mean(vectors, axis=0)
        print(f"✅ Baseline captured from {len(texts)} inputs")
        print(f"   Baseline vector shape: {self.baseline_vector.shape}")

    def check_drift(self, current_texts: list) -> dict:
        """Current session ka drift check karo"""
        if self.baseline_vector is None:
            return {"error": "Baseline not set yet!"}

        self.session_count += 1

        # Current session ka vector banao
        vectors = [simple_embed(t) for t in current_texts]
        current_vector = np.mean(vectors, axis=0)

        # Drift calculate karo
        drift_score = calculate_drift(self.baseline_vector, current_vector)
        status = get_drift_status(drift_score)

        # History mein save karo
        entry = {
            "session": self.session_count,
            "timestamp": datetime.now().isoformat(),
            "drift_score": drift_score,
            "status": status["status"],
            "inputs_analyzed": len(current_texts)
        }
        self.drift_history.append(entry)

        return {
            "drift_score": drift_score,
            "status": status,
            "entry": entry
        }

    def save_history(self):
        """Drift history save karo"""
        with open("drift_history.json", "w") as f:
            json.dump(self.drift_history, f, indent=2)
        print(f"💾 Drift history saved ({len(self.drift_history)} sessions)")

# ── TEST KARO ────────────────────────────────────────
if __name__ == "__main__":
    print("🧠 MemoryOS — Drift Detection Engine")
    print("="*40)

    detector = DriftDetector()

    # Step 1: Baseline set karo (normal AI questions)
    print("\n📍 Setting baseline (normal behavior)...")
    baseline_inputs = [
        "what is machine learning",
        "what is deep learning",
        "what is neural network",
        "what is python programming"
    ]
    detector.set_baseline(baseline_inputs)

    # Step 2: Similar session test (low drift expected)
    print("\n🔍 Session 1: Similar topics (low drift expected)...")
    session1 = [
        "explain artificial intelligence",
        "what is data science",
        "how does machine learning work"
    ]
    result1 = detector.check_drift(session1)
    print(f"   Drift Score: {result1['drift_score']}")
    print(f"   Status: {result1['status']['emoji']} {result1['status']['status']}")
    print(f"   {result1['status']['message']}")

    # Step 3: Very different session (high drift expected)
    print("\n🔍 Session 2: Very different topics (high drift expected)...")
    session2 = [
        "what is cricket score today",
        "recipe for biryani",
        "best movies 2026",
        "weather in mumbai"
    ]
    result2 = detector.check_drift(session2)
    print(f"   Drift Score: {result2['drift_score']}")
    print(f"   Status: {result2['status']['emoji']} {result2['status']['status']}")
    print(f"   {result2['status']['message']}")

    # Step 4: History save karo
    detector.save_history()

    print("\n✅ Day 2 Complete!")
    print("   drift_history.json file check karo!")