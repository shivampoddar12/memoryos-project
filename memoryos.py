# memoryos.py — MemoryOS Complete System — Day 5
# Sab modules ek jagah!

import numpy as np
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 MemoryOS — Complete System Loading...")

# ══════════════════════════════════════════
# 1. EMBEDDER
# ══════════════════════════════════════════
class Embedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500, ngram_range=(1,2), stop_words='english'
        )
        self.fitted = False

    def fit_and_embed(self, texts: list) -> np.ndarray:
        self.vectorizer.fit(texts)
        self.fitted = True
        vectors = self.vectorizer.transform(texts).toarray()
        return np.mean(vectors, axis=0)

    def embed(self, texts: list) -> np.ndarray:
        if not self.fitted:
            return self.fit_and_embed(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        return np.mean(vectors, axis=0)

# ══════════════════════════════════════════
# 2. DRIFT DETECTOR
# ══════════════════════════════════════════
class DriftDetector:
    def __init__(self, threshold: float = 0.45):
        self.embedder = Embedder()
        self.baseline = None
        self.threshold = threshold
        self.history = []

    def set_baseline(self, texts: list):
        self.baseline = self.embedder.fit_and_embed(texts)
        print(f"   ✅ Baseline set from {len(texts)} texts")

    def check(self, texts: list) -> dict:
        current = self.embedder.embed(texts)
        sim = cosine_similarity([self.baseline], [current])[0][0]
        score = round(1.0 - float(sim), 4)

        if score < 0.3:
            status, emoji = "HEALTHY", "🟢"
        elif score < 0.5:
            status, emoji = "WARNING", "🟡"
        elif score < 0.7:
            status, emoji = "CRITICAL", "🔴"
        else:
            status, emoji = "DANGER", "🚨"

        result = {
            "score": score, "status": status, "emoji": emoji,
            "heal_needed": score >= self.threshold,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(result)
        return result

# ══════════════════════════════════════════
# 3. AUTO HEAL
# ══════════════════════════════════════════
class AutoHealer:
    def __init__(self):
        self.heal_count = 0
        self.heal_log = []

    def heal(self, memory_log: list, drift_score: float) -> list:
        self.heal_count += 1
        print(f"\n   💊 Auto-Heal #{self.heal_count} starting...")

        # Relevance score calculate karo
        scored = []
        for i, entry in enumerate(memory_log):
            age = len(memory_log) - i
            decay = np.exp(-0.1 * age)
            relevance = decay
            scored.append((relevance, entry))

        # Sort by relevance
        scored.sort(reverse=True)

        # Top 3 rakho
        top_k = min(3, len(scored))
        healed = [entry for _, entry in scored[:top_k]]

        self.heal_log.append({
            "heal_number": self.heal_count,
            "timestamp": datetime.now().isoformat(),
            "drift_score": drift_score,
            "memories_before": len(memory_log),
            "memories_after": len(healed)
        })

        print(f"   ✅ Healed! {len(memory_log)} → {len(healed)} memories")
        return healed

# ══════════════════════════════════════════
# 4. AGENT STATE
# ══════════════════════════════════════════
class AgentState(TypedDict):
    messages: List[str]
    memory_log: List[dict]
    session_count: int
    last_response: str
    drift_score: float
    health_status: str

# ══════════════════════════════════════════
# 5. MEMORYOS — MAIN SYSTEM
# ══════════════════════════════════════════
class MemoryOS:
    def __init__(self):
        self.detector = DriftDetector(threshold=0.45)
        self.healer = AutoHealer()
        self.session_count = 0
        self.is_initialized = False
        print("✅ MemoryOS initialized!")

    def initialize(self, baseline_texts: list):
        """System ko baseline se setup karo"""
        print("\n📍 Setting baseline...")
        self.detector.set_baseline(baseline_texts)
        self.is_initialized = True
        print("✅ System ready!")

    def process(self, user_input: str, memory_log: list) -> dict:
        """Har message pe yeh run hoga"""
        self.session_count += 1

        # Simple response (bina OpenAI ke)
        response = f"[Session {self.session_count}] Received: '{user_input}'"

        # Memory mein add karo
        entry = {
            "session": self.session_count,
            "time": datetime.now().isoformat(),
            "input": user_input,
            "response": response
        }
        memory_log = memory_log + [entry]

        # Drift check karo (session 2 se)
        drift_result = {"score": 0.0, "status": "HEALTHY",
                       "emoji": "🟢", "heal_needed": False}

        if self.session_count >= 2 and self.is_initialized:
            recent_inputs = [e["input"] for e in memory_log[-3:]]
            drift_result = self.detector.check(recent_inputs)

            print(f"\n{'='*40}")
            print(f"🔍 Session {self.session_count} Health Check")
            print(f"   Drift Score: {drift_result['score']}")
            print(f"   Status: {drift_result['emoji']} {drift_result['status']}")

            # Auto heal zarurat hai?
            if drift_result["heal_needed"]:
                memory_log = self.healer.heal(memory_log, drift_result["score"])

        return {
            "response": response,
            "memory_log": memory_log,
            "drift_score": drift_result["score"],
            "health_status": drift_result["status"]
        }

    def save_all(self, memory_log: list):
        """Sab data save karo"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_sessions": self.session_count,
            "drift_history": self.detector.history,
            "heal_log": self.healer.heal_log,
            "final_memory": memory_log
        }
        with open("memoryos_data.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 All data saved to memoryos_data.json")
        print(f"   Total sessions: {self.session_count}")
        print(f"   Total heals: {self.healer.heal_count}")
        print(f"   Drift checks: {len(self.detector.history)}")


# ══════════════════════════════════════════
# 6. MAIN — INTERACTIVE LOOP
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*45)
    print("🤖 MemoryOS — Complete System")
    print("="*45)

    # Initialize
    os_system = MemoryOS()
    os_system.initialize([
        "what is machine learning",
        "what is deep learning",
        "what is neural network",
        "what is python programming"
    ])

    memory_log = []

    print("\n💬 Commands:")
    print("   'status' — health check dekho")
    print("   'memory' — memory log dekho")
    print("   'quit'   — band karo")
    print("\nShuru karo! Kuch bhi likho:\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        elif user_input.lower() == "quit":
            os_system.save_all(memory_log)
            print("\n👋 MemoryOS band ho gaya!")
            break

        elif user_input.lower() == "status":
            print(f"\n📊 MemoryOS Status:")
            print(f"   Sessions: {os_system.session_count}")
            print(f"   Memories: {len(memory_log)}")
            print(f"   Heals done: {os_system.healer.heal_count}")
            if os_system.detector.history:
                last = os_system.detector.history[-1]
                print(f"   Last drift: {last['score']} {last['emoji']}")
            continue

        elif user_input.lower() == "memory":
            print(f"\n🧠 Memory Log ({len(memory_log)} entries):")
            for m in memory_log[-5:]:
                print(f"   [{m['session']}] {m['input'][:40]}")
            continue

        # Process karo
        result = os_system.process(user_input, memory_log)
        memory_log = result["memory_log"]

        print(f"🤖 {result['response']}")
        print(f"   [{result['health_status']}]")